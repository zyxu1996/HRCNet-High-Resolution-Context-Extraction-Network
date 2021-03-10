import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
from hrcnet import PoseHighResolutionNet
from dataset import potsdam
import setproctitle
import time
import sklearn.metrics as metric
from loss import SE_loss, Edge_loss
import logging
logging.basicConfig(filename='hrnet_v11.log', level=logging.INFO)

# 0=impervious surfacescd
# 1=building
# 2=low vegetation
# 3=tree
# 4=car
# 5=background

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ.setdefault('RANK', '0')
# os.environ.setdefault('WORLD_SIZE', '1')
# os.environ.setdefault('MASTER_ADDR', '202.204.54.125')
# os.environ.setdefault('MASTER_PORT', '29555')

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss, edge_loss1, se_loss1):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss
    self.edge_loss = edge_loss1
    self.se_loss = se_loss1

  def forward(self, inputs, labels, edge=None, train=True):
    output, edge_out, se_out = self.model(inputs)
    if train:
        seg_loss = self.loss(output, labels)
        edge_loss = self.edge_loss(edge_out, edge)
        se_loss = self.se_loss(se_out, labels)

        loss = seg_loss + 0.9*edge_loss + 0.2 * se_loss
        return torch.unsqueeze(loss, 0)
    else:
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

        # "hrnet32_light"
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


def get_params():
    pa = params()
    return pa


def get_model(args):
    model = PoseHighResolutionNet(args)
    return model


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


def train():
    args = get_params()

    args2 = parse_args()

    dataset_dir = "./dataset"

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    distributed = True
    device = torch.device(('cuda:{}').format(args2.local_rank))
    PoseHighResolutionNet = get_model(args)

    if distributed:
        torch.cuda.set_device(args2.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    potsdam_train = potsdam(base_dir=dataset_dir, train=True)
    if distributed:
        train_sampler = DistributedSampler(potsdam_train)
    else:
        train_sampler = None
    dataloader_train = DataLoader(
        potsdam_train,
        batch_size=args.TRAIN_BATCH_SIZE_PER_GPU,
        shuffle=True and train_sampler is None,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    potsdam_val = potsdam(base_dir=dataset_dir, train=False)
    if distributed:
        val_sampler = DistributedSampler(potsdam_val)
    else:
        val_sampler = None
    dataloader_val = DataLoader(
        potsdam_val,
        batch_size=args.VAL_BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler)

    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    edge_criterion = Edge_loss()
    se_criterion = SE_loss()

    # print(PoseHighResolutionNet)
    # output = sys.stdout
    # outputfile = open("hrnet_structure.txt", "a")
    # sys.stdout = outputfile

    PoseHighResolutionNet = FullModel(PoseHighResolutionNet, seg_criterion, edge_criterion, se_criterion)
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

    # PoseHighResolutionNet.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('model/hrnetv2_w48_imagenet_pretrained.pth').items()})
    # PoseHighResolutionNet.load_state_dict(torch.load('model/hrnetv2_w48_imagenet_pretrained.pth'))

    start = time.clock()
    end_epoch = 600
    lr = args.learning_rate
    miou = [0]
    best_miou = 0.1
    last_epoch = 400
    test_epoch = end_epoch - 50
    ave_loss = AverageMeter()
    world_size = get_world_size()
    reduced_loss = [0]

    model_state_file = "model/checkpoint_hrnet32_cut_384_72.pkl.tar"
    if os.path.isfile(model_state_file):
        logging.info("=> loading checkpoint '{}'".format(model_state_file))
        checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
        best_miou = checkpoint['best_mIoU']
        last_epoch = checkpoint['epoch']
        PoseHighResolutionNet.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(
            model_state_file, checkpoint['epoch']))

    for epoch in range(last_epoch, end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        PoseHighResolutionNet.train()
        setproctitle.setproctitle("xzy:" + str(epoch) + "/" + "{}".format(end_epoch))
        for i, sample in enumerate(dataloader_train):
            images, labels, edge = sample['image'], sample['label'], sample['edge']
            images, labels, edge = images.to(device), labels.to(device), edge.to(device)
            labels = labels.long().squeeze(1)
            edge = edge.long().squeeze(1)
            losses = PoseHighResolutionNet(images, labels, edge)
            loss = losses.mean()

            ave_loss.update(loss.item())

            lr = adjust_learning_rate(optimizer,
                                      args.learning_rate,
                                      end_epoch * len(dataloader_train),
                                      i + epoch * len(dataloader_train))
            if i % 50 == 0:
                reduced_loss[0] = ave_loss.average()
                print_loss = reduce_tensor(torch.from_numpy(np.array(reduced_loss)).to(device)).cpu()[0] / world_size
                if args2.local_rank == 0:

                    time_cost = time.clock() - start
                    print("epoch:[{}/{}], iter:[{}/{}], loss:{}, time:{}, lr:{}, best_miou:{}".format(epoch,end_epoch,i,len(dataloader_train),print_loss,time_cost,lr,best_miou))
                    logging.info(
                        "epoch:[{}/{}], iter:[{}/{}], loss:{}, time:{}, lr:{}, best_miou:{}, miou:{}".format(epoch, end_epoch, i,
                                                                                                len(dataloader_train),
                                                                                                print_loss, time_cost, lr,
                                                                                                best_miou, miou[0]))
            PoseHighResolutionNet.zero_grad()
            loss.backward()
            optimizer.step()
            start = time.clock()

        if epoch > test_epoch:
            OA = validate(dataloader_val, device, PoseHighResolutionNet)
            miou = reduce_tensor(OA).cpu()

        if args2.local_rank == 0:
            print(miou)

            if epoch % 100 == 0 and epoch != 0:
                torch.save(PoseHighResolutionNet.state_dict(),
                           'model/edge1_se_origin_hrnet48_3463__cut_acfpn3463_384_72_sem_gcnet_stage4_xzy_{0}_{1}.pkl'.format(epoch, args.learning_rate))

            if miou[0] >= best_miou:
                best_miou = miou[0]
                torch.save(PoseHighResolutionNet.state_dict(),
                           'model/edge1_se_hrnet48_3463_cut_acfpn3463_384_72_sem_gcnet_stage4_best_result_{}.pkl'.format(epoch))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_miou,
                'state_dict': PoseHighResolutionNet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'model/edge1_se_checkpoint_hrnet48_3463_cut_acfpn3463_384_72_sem_gcnet_stage4.pkl.tar')  # checkpoint_hrnet32_cut_384_72.pkl.tar'
    torch.save(PoseHighResolutionNet.state_dict(),
                   'model/edge1_se_origin_hrnet48_3463_cut_acfpn3463_384_72_sem_gcnet_stage4_xzy_{0}_{1}.pkl'.format(end_epoch, args.learning_rate))


def adjust_learning_rate(optimizer, base_lr, max_iters,
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def validate(dataloader_val, device, PoseHighResolutionNet):
    PoseHighResolutionNet.eval()
    OA = [0]
    n = 0
    with torch.no_grad():
        for i, sample in enumerate(dataloader_val):
            images, labels = sample['image'], sample['label']
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)
            logits = PoseHighResolutionNet(images, labels, train=False)
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            for j in range(logits.shape[0]):
                n += 1
                prediction = logits[j]
                prediction = np.reshape(prediction, (384 * 384))
                label = labels[j]
                label = np.reshape(label, (384 * 384))
                oa = metric.accuracy_score(label, prediction)
                OA = OA + oa
    OA = OA/(2*n)
    print("OA:{}".format(OA))
    OA = torch.from_numpy(OA).to(device)

    return OA


if __name__ == '__main__':
    train()




