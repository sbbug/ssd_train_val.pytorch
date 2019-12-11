from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from trainers.trainer import Trainer
import argparse
from logger import make_logger
from trainers import weights_init


def train(opt):
    train_dataset = VOCDetection(
        transform=SSDAugmentation(opt.DATASETS.MIN_DIM, opt.DATASETS.MEANS), opt=opt)

    test_dataset = VOCDetection(['test'],
                                BaseTransform(300, opt.DATASETS.MEANS),
                                VOCAnnotationTransform(),opt=opt)

    ssd_net = build_ssd('train', opt.DATASETS.MIN_DIM, opt.DATASETS.NUM_CLS)
    net = ssd_net
    # logger
    logger = make_logger("project", opt.OUTPUT_DIR, 'log')

    if len(opt.DEVICE_ID) > 1:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if opt.MODEL.RESUM:
        print('Resuming training, loading {}...'.format(opt.MODEL.RESUM))
        ssd_net.load_weights(opt.MODEL.RESUM)
    else:
        vgg_weights = torch.load(os.path.join(opt.MODEL.BACKBONE_WEIGHTS, opt.MODEL.BACKBONE))
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if opt.DEVICE:
        net = net.cuda()

    if not opt.MODEL.RESUM:
        print('Initializing backbone_weights...')
        # initialize newly added layers' backbone_weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=opt.SOLVER.BASE_LR, momentum=opt.SOLVER.MOMENTUM,
                          weight_decay=opt.SOLVER.WEIGHT_DECAY)
    criterion = MultiBoxLoss(opt.DATASETS.NUM_CLS, 0.5, True, 0, True, 3, 0.5,
                             False, opt.DEVICE)
    epoch_size = len(train_dataset) // opt.DATALOADER.BATCH_SIZE

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=opt.DATALOADER.BATCH_SIZE,
                                   num_workers=opt.DATALOADER.NUM_WORKERS,
                                   shuffle=True,
                                   collate_fn=detection_collate,
                                   pin_memory=True
                                   )
    # device = torch.device("cuda")
    trainer = Trainer(
        net,
        optimizer,
        criterion,
        logger,
        device=None,
        scheduler=None
    )
    trainer.run(
        opt=opt,
        train_loader=train_loader,  # dataloader
        test_dataset=test_dataset,  # dataset
        epoch_size=epoch_size
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Weather Classification Project.")
    parser.add_argument('--config', default='./configs/sample.yaml')
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    if opt.DEVICE:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.DEVICE_ID
        num_gpus = len(opt.DEVICE_ID.split(','))


    if torch.cuda.is_available():
        if opt.DEVICE:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not opt.DEVICE:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(os.path.join(opt.OUTPUT_DIR)):
        os.mkdir(os.path.join(opt.OUTPUT_DIR))
    if not os.path.exists(os.path.join(opt.OUTPUT_DIR, "backbone_weights")):
        os.mkdir(os.path.join(opt.OUTPUT_DIR, "backbone_weights"))

    train(opt)
