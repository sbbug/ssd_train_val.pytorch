import torch
from utils.show_visdom import update_vis_plot, create_vis_plot
from scheduler.warmup_scheduler import adjust_learning_rate
from torch.autograd import Variable
import time
from eval import test_net
from data import BaseTransform


class Trainer:

    def __init__(self, model, optimizer, criterion, logger, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.device = device
        self.scheduler = scheduler

    def run(self, opt, train_loader, test_dataset, epoch_size, val_loader=None, eval_period=None):

        self.logger.info('Start at Epoch[{}]'.format(opt.SOLVER.START_EPOCH))

        loc_loss = 0
        conf_loss = 0
        epoch = 0
        step_index = 0

        best_map = 0.0

        if opt.SHOW.VISDOM:
            import visdom
            viz = visdom.Visdom()
            vis_title = 'SSD.PyTorch on ' + opt.DATASETS.NAME
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend, viz)
            epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend, viz)

        batch_iterator = iter(train_loader)
        self.model.train()

        for iteration in range(opt.SOLVER.START_EPOCH, opt.SOLVER.MAX_EPOCHS):
            if opt.SHOW.VISDOM and iteration != 0 and (iteration % epoch_size == 0):
                update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                                'append', viz, epoch_size)

                loc_loss = 0
                conf_loss = 0
                epoch += 1

            if iteration in opt.SCHEDULER.STEP:
                step_index += 1
                adjust_learning_rate(self.optimizer, opt.SCHEDULER.GAMMA, step_index, self.optimizer.defaults['lr'])

            # load train data
            try:
                images, targets = next(batch_iterator)  # 进行异常捕获
            except StopIteration:
                batch_iterator = iter(train_loader)
                images, targets = next(batch_iterator)

            if opt.DEVICE:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]
            # forward
            t0 = time.time()
            out = self.model(images)
            # backprop
            self.optimizer.zero_grad()

            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            self.optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if iteration % 1 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            if opt.SHOW.VISDOM:
                update_vis_plot(iteration, loss_l.item(), loss_c.data.item(), iter_plot, epoch_plot, 'append', viz, )

            if iteration != 0 and iteration % opt.MODEL.SAVE_MODEL_FRE == 0:
                print('Saving state, iter:', iteration)
                torch.save(self.model.state_dict(), opt.OUTPUT_DIR +
                           repr(iteration) + '.pth')
            # 进行测试
            if (1 + iteration) % opt.MODEL.VAL_GAP == 0:

                print("valing....")
                self.model.eval()
                self.model.phase = 'test'
                map = test_net(opt,
                               self.model,
                               test_dataset,
                               BaseTransform(self.model.size, opt.DATASETS.MEANS),
                               opt.MODEL.TOP_K,
                               300
                               )
                self.model.train()
                self.model.phase = 'train'
                self.logger.info("map is {}".format(map))
                if map > best_map:
                    best_map = map
                    torch.save(self.model.state_dict(),
                               opt.OUTPUT_DIR + '/' + opt.DATASETS.NAME + '.pth')

        print("best map is {}".format(best_map))
        self.logger.info('Epoch[{}] Finished.\n'.format(epoch))
