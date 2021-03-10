import logging
import torch
import torch.nn as nn
import torch._utils
from torch.nn import functional as F

BN_MOMENTUM = 0.01
BatchNorm2d = nn.BatchNorm2d
logger = logging.getLogger(__name__)
act_fn = nn.ReLU(inplace=True)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=1./4.,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


def conv1x1(in_planes, out_planes, stride=1):
    conv1x1 = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
        act_fn
    )
    return conv1x1


def dilaconv(in_planes, out_planes, stride=1, dilation=1, padding=0):
    dilaconv = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  bias=False, dilation=dilation, padding=padding),
        BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
        act_fn
    )
    return dilaconv


class CEM(nn.Module):
    def __init__(self, planes):
        super(CEM, self).__init__()

        self.conv1 = conv1x1(planes, planes // 4)
        self.dilaconv1 = dilaconv(planes // 4, planes // 8, dilation=2, padding=2)
        self.conv2 = conv1x1(planes + planes // 8, planes // 4)
        self.dilaconv2 = dilaconv(planes // 4, planes // 8, dilation=3, padding=3)
        self.conv3 = conv1x1(planes + planes // 4, planes // 4)
        self.dilaconv3 = dilaconv(planes // 4, planes // 8, dilation=4, padding=4)
        self.conv4 = conv1x1(planes + planes // 4 + planes // 8, planes // 4)
        self.dilaconv4 = dilaconv(planes // 4, planes // 8, dilation=5, padding=5)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv6 = conv1x1(planes, planes // 8)
        self.conv7 = conv1x1(5 * planes // 8, planes // 8)

    def forward(self, x):
        _, _, h, w = x.size()
        y = x
        x1 = self.conv1(x)
        x1 = self.dilaconv1(x1)
        x1_cat = torch.cat((x, x1), dim=1)

        x2 = self.conv2(x1_cat)
        x2 = self.dilaconv2(x2)
        x2_cat = torch.cat((x1_cat, x2), dim=1)

        x3 = self.conv3(x2_cat)
        x3 = self.dilaconv3(x3)
        x3_cat = torch.cat((x2_cat, x3), dim=1)

        x4 = self.conv4(x3_cat)
        x4 = self.dilaconv4(x4)

        y = self.global_pool(y)
        y = self.conv6(y)
        y = F.interpolate(y, size=(h, w), mode="bilinear")

        concat = torch.cat((x1, x2, x3, x4, y), dim=1)
        final = self.conv7(concat)

        return final


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            act_fn,
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, stage4=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.act_fn = act_fn
        self.conv2 = conv3x3(planes, planes)

        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stage4 = stage4
        if self.stage4:
            self.se = SELayer(planes, reduction)
            self.gcblock= ContextBlock(inplanes=inplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.stage4:
            residual = self.gcblock(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.stage4:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fn(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, stage4=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)

        self.bn3 = BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.act_fn = act_fn
        self.stage4 = stage4
        if self.stage4:
            self.se = SELayer(planes * 4, reduction)
            self.gcblock = ContextBlock(inplanes=inplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.stage4:
            residual = self.gcblock(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_fn(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.stage4:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fn(out)
        return out


class _BoundaryAwareness(nn.Module):
    """Edge awareness module"""

    def __init__(self, in_fea=[32, 64], mid_fea=48, out_fea=2):
        super(_BoundaryAwareness, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], in_fea[0], kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            BatchNorm2d(in_fea[0], momentum=BN_MOMENTUM),
            act_fn,
            nn.Conv2d(in_fea[0], mid_fea, 1, 1, 0, bias=False),
            BatchNorm2d(mid_fea, momentum=BN_MOMENTUM)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], in_fea[1], kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            BatchNorm2d(in_fea[1], momentum=BN_MOMENTUM),
            act_fn,
            nn.Conv2d(in_fea[1], mid_fea, 1, 1, 0, bias=False),
            BatchNorm2d(mid_fea, momentum=BN_MOMENTUM)
        )
        self.conv3 = nn.Conv2d(
            mid_fea * 2, out_fea, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)  # (128)
        edge2_fea = self.conv2(x2)  # (64)

        edge1_fea = F.interpolate(edge1_fea, size=(
            h, w), mode='bilinear')
        edge2_fea = F.interpolate(edge2_fea, size=(
            h, w), mode='bilinear')
        edge_fea = torch.cat([edge1_fea, edge2_fea], dim=1)
        edge = self.conv3(edge_fea)

        return edge, edge_fea


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.act_fn = act_fn

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            if i < num_blocks[branch_index] - 1:
                layers.append(
                    block(
                        self.num_inchannels[branch_index],
                        num_channels[branch_index]
                    )
                )
            else:
                layers.append(
                    block(
                        self.num_inchannels[branch_index],
                        num_channels[branch_index], stage4=True
                    )
                )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                    act_fn
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.act_fn(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.nonlocal1 = ContextBlock(inplanes=64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.nonlocal2 = ContextBlock(inplanes=64)
        self.act_fn = act_fn
        self.layer1 = self._make_layer(Bottleneck, 64, 3)

        self.stage2_cfg = cfg.STAGE2
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg.STAGE3
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg.STAGE4
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        channels = self.stage4_cfg['NUM_CHANNELS']
        last_inp_channels = channels[0] + channels[1]
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            act_fn,
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=cfg.number_of_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        # Top layer
        self.toplayer = nn.Conv2d(channels[3], channels[0], kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.CEM = CEM(channels[3])
        self.se = nn.Sequential(
            SELayer(channels[3]),
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(channels[3], 6),
            nn.Sigmoid(),
        )

        # Lateral layers
        self.latlayer1 = nn.Conv2d(channels[2], channels[0], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(channels[1], channels[0], kernel_size=1, stride=1, padding=0)

        self.edge = _BoundaryAwareness(in_fea=[channels[0], 64], mid_fea=channels[0])

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            act_fn
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            act_fn
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i < blocks - 1:
                layers.append(block(self.inplanes, planes))
            else:
                layers.append(block(self.inplanes, planes, stage4=True))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = x / 255
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        x = self.nonlocal1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_fn(x)

        x = self.nonlocal2(x)
        edge_y = x
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        y3 = self.toplayer(y_list[3])
        y3 = self.CEM(y_list[3]) + y3

        se = self.se[0](y_list[3])
        se = self.se[1](se)
        se = se.view(se.size(0), -1)
        se = self.se[2](se)
        se = self.se[3](se)

        y2 = F.interpolate(y3, scale_factor=2, mode='bilinear') + self.latlayer1(y_list[2])
        y1 = F.interpolate(y2, scale_factor=2, mode='bilinear') + self.latlayer2(y_list[1])
        edge_x, edge_fuse = self.edge(y_list[0], edge_y)
        y0 = torch.cat((F.interpolate(y1, scale_factor=2, mode='bilinear'), edge_fuse), dim=1)
        x = self.final_layer(y0)
        x = x * (se.view(-1, 6, 1, 1))
        x = F.interpolate(x, scale_factor=4, mode="bilinear")

        edge_x = F.interpolate(edge_x, scale_factor=4, mode="bilinear")
        return x, edge_x, se

