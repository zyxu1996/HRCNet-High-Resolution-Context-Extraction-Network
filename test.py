import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from hrcnet import PoseHighResolutionNet
from dataset import potsdam
from loss import Edge_loss
import time
from scipy import io
import logging
logging.basicConfig(filename='hrnet_v11.log', level=logging.INFO)

# 0=impervious surfacescd
# 1=building
# 2=low vegetation
# 3=tree
# 4=car
# 5=background


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '202.204.54.125')
os.environ.setdefault('MASTER_PORT', '29555')

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss, edge_loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss
    self.edge_loss = edge_loss

  def forward(self, inputs, labels):
    output, edge_out, se_out = self.model(inputs)
    return output


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class params():
    def __init__(self):
        self.number_of_classes = 6
        self.TRAIN_BATCH_SIZE_PER_GPU = 8
        self.VAL_BATCH_SIZE_PER_GPU = 16
        self.learning_rate = 0.01
        "hrnet48"
        self.STAGE2 = {'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'NUM_BLOCKS': [4,4],
                        'NUM_CHANNELS': [48,96],
                        'BLOCK':'BASIC',
                        'FUSE_METHOD': 'SUM'}
        self.STAGE3 = {'NUM_MODULES': 4,
                       'NUM_BRANCHES': 3,
                       'NUM_BLOCKS': [6, 6, 6],
                       'NUM_CHANNELS': [48, 96, 192],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
        self.STAGE4 = {'NUM_MODULES': 3,
                       'NUM_BRANCHES': 4,
                       'NUM_BLOCKS': [3, 3, 3, 3],
                       'NUM_CHANNELS': [48, 96, 192, 384],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
        # "hrnet32"
        # self.STAGE2 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 2,
        #                'NUM_BLOCKS': [4, 4],
        #                'NUM_CHANNELS': [32, 64],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE3 = {'NUM_MODULES': 4,
        #                'NUM_BRANCHES': 3,
        #                'NUM_BLOCKS': [4, 4, 4],
        #                'NUM_CHANNELS': [32, 64, 128],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE4 = {'NUM_MODULES': 3,
        #                'NUM_BRANCHES': 4,
        #                'NUM_BLOCKS': [4, 4, 4, 4],
        #                'NUM_CHANNELS': [32, 64, 128, 256],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # "hrnet32_cut"
        # self.STAGE2 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 2,
        #                'NUM_BLOCKS': [4, 4],
        #                'NUM_CHANNELS': [32, 64],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE3 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 3,
        #                'NUM_BLOCKS': [6, 6, 6],
        #                'NUM_CHANNELS': [32, 64, 128],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}
        # self.STAGE4 = {'NUM_MODULES': 1,
        #                'NUM_BRANCHES': 4,
        #                'NUM_BLOCKS': [3, 3, 3, 3],
        #                'NUM_CHANNELS': [32, 64, 128, 256],
        #                'BLOCK': 'BASIC',
        #                'FUSE_METHOD': 'SUM'}


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


args = params()

args2 = parse_args()

dataset_dir = "../dataset"

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True
distributed = True
device = torch.device(('cuda:{}').format(args2.local_rank))
PoseHighResolutionNet = PoseHighResolutionNet(args)

# flops, params = profile(PoseHighResolutionNet, (1, 3, 384, 384))
# print('flops:{}G'.format(flops/1e9), 'params:{}M'.format(params/(1024*1024)))

if distributed:
    torch.cuda.set_device(args2.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
    )

potsdam_test = potsdam(base_dir=dataset_dir, train=False)
val_sampler = None
dataloader_test = DataLoader(
    potsdam_test,
    batch_size=args.VAL_BATCH_SIZE_PER_GPU,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=val_sampler)

seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
edge_criterion = Edge_loss()

PoseHighResolutionNet = FullModel(PoseHighResolutionNet, seg_criterion, edge_criterion)
PoseHighResolutionNet = nn.SyncBatchNorm.convert_sync_batchnorm(PoseHighResolutionNet)
PoseHighResolutionNet = PoseHighResolutionNet.to(device)
PoseHighResolutionNet = nn.parallel.DistributedDataParallel(
    PoseHighResolutionNet, device_ids=[args2.local_rank], output_device=args2.local_rank)

optimizer = torch.optim.SGD([{'params':
                                  filter(lambda p: p.requires_grad,
                                         PoseHighResolutionNet.parameters()),
                                  'lr': args.learning_rate}],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=0.0005,
                                nesterov=False,
                                )


def test():
    PoseHighResolutionNet.eval()
    with torch.no_grad():
        model_state_file = "model/edge1_se_hrnet48_3463_cut_acfpn3463_384_72_sem_gcnet_stage4_best_result_397.pkl"
        if os.path.isfile(model_state_file):
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            PoseHighResolutionNet.load_state_dict(checkpoint)
        n = 0
        for i, sample in enumerate(dataloader_test):
            images, labels = sample['image'], sample['label']
            images, labels = images.cuda(), labels.cuda()
            labels = labels.long().squeeze(1)
            start = time.clock()
            logits = PoseHighResolutionNet(images, labels)
            time_cost = time.clock() - start
            print("time_cost:{}".format(time_cost))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            for j in range(logits.shape[0]):
                n += 1
                io.savemat("network_output/" + str(n) + ".mat", {'network_output': logits[j]})


if __name__ == '__main__':
    test()





