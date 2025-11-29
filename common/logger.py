r""" Logging during training/testing """
import datetime
import logging
import os

try:
    from tensorboardX import SummaryWriter
except Exception:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'isaid':
            self.nclass = 15
        elif self.benchmark == 'dlrsd':
            self.nclass = 15
        elif self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        # Accept dict or namespace for args
        from types import SimpleNamespace
        if isinstance(args, dict):
            args = SimpleNamespace(**args)

        # Provide some safe defaults if missing
        if not hasattr(args, 'logpath'):
            args.logpath = ''
        if not hasattr(args, 'load'):
            args.load = ''
        if not hasattr(args, 'benchmark'):
            args.benchmark = 'pascal'

        # build log name
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        if training:
            logname = args.logpath if args.logpath else logtime
        else:
            # safe guard for args.load possibly empty or not a path-like string
            try:
                load_part = args.load.split('/')[-2].split('.')[0]
                logname = '_TEST_' + load_part + logtime
            except Exception:
                logname = '_TEST_' + logtime

        # create directory under logs/
        cls.logdir = os.path.join('logs', logname)
        os.makedirs(cls.logdir, exist_ok=True)

        # configure python logging to write to file inside that dir
        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logdir, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer (safe fallback if not available)
        try:
            if SummaryWriter is not None:
                cls.tbd_writer = SummaryWriter(os.path.join(cls.logdir, 'tbd', 'runs'))
            else:
                cls.tbd_writer = None
        except Exception:
            cls.tbd_writer = None

        # store benchmark and logpath for other methods
        cls.benchmark = args.benchmark
        cls.logpath = cls.logdir  # keep backward-compatible attribute name

        # Log arguments (handle namespace or dict)
        logging.info('\n:=========== Few-shot Seg. with HSNet ===========')
        if hasattr(args, '__dict__'):
            items = args.__dict__.items()
        else:
            # fallback: iterate attributes
            items = [(k, getattr(args, k)) for k in dir(args) if not k.startswith('_')]
        for arg_key, arg_val in items:
            logging.info('| %20s: %-24s' % (str(arg_key), str(arg_val)))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))
