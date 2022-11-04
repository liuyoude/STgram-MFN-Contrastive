import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
import yaml
import os

from data_func.dataset import *
from model import *
from trainer import *

from model import MobileFaceNet
from simclrv2 import SimCLRv2, SimCLRv2_ft

config_path = './config.yaml'
with open(config_path) as f:
    params = yaml.safe_load(f)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='STgram-MFN-Contrastive')
for key, value in params.items():
    parser.add_argument(f'--{key}', default=value, type=type(value))


def pretrain(args):
    simclr_net = SimCLRv2(num_class=args.num_classes)
    gpu_id = args.device_ids[0]
    if torch.cuda.is_available():
        args.gpu_index = gpu_id
        args.device = torch.device(f'cuda:{gpu_id}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    con_dataset = WavMelClassifierDataset(args.train_dirs, args.sr, args.ID_factor,
                                          pattern='uniform' if args.pretrain_loss == 'ntxent' else 'random')
    train_clf_dataset = con_dataset.get_dataset(n_mels=args.n_mels,
                                                n_fft=args.n_fft,
                                                hop_length=args.hop_length,
                                                win_length=args.win_length,
                                                power=args.power)
    train_clf_loader = torch.utils.data.DataLoader(
        train_clf_dataset, batch_size=args.con_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    optimizer = torch.optim.Adam(simclr_net.parameters(), lr=float(args.con_lr))
    with torch.cuda.device(args.gpu_index):
        args.dp = True if len(args.device_ids) > 1 else False
        if len(args.device_ids) > 1:
            simclr_net = torch.nn.DataParallel(simclr_net, device_ids=args.device_ids)
        trainer = CLRTrainer(data_dir=args.data_dir,
                             id_fctor=args.ID_factor,
                             net=simclr_net,
                             optimizer=optimizer,
                             scheduler=None,
                             args=args)
        trainer.train(train_clf_loader)


def finetune(args, epoch=100):
    pretrain_model_path = os.path.join(args.model_dir, args.version, 'pretrain', f'checkpoint_{epoch:04d}.pth.tar') \
        if args.pretrain else None
    net = SimCLRv2(pretrained_weights=pretrain_model_path)
    ft_net = SimCLRv2_ft(net, args.num_classes, use_arcface=args.use_arcface,
                         pretrain=args.pretrain, m=args.m, s=args.s, sub=args.sub_center)
    gpu_id = args.ft_ids[0]
    if torch.cuda.is_available():
        args.gpu_index = gpu_id
        args.device = torch.device(f'cuda:{gpu_id}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    clf_dataset = WavMelClassifierDataset(args.train_dirs, args.sr, args.ID_factor, pattern='random')
    train_clf_dataset = clf_dataset.get_dataset(n_mels=args.n_mels,
                                                n_fft=args.n_fft,
                                                hop_length=args.hop_length,
                                                win_length=args.win_length,
                                                power=args.power)
    train_clf_loader = torch.utils.data.DataLoader(
        train_clf_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    optimizer = torch.optim.Adam(ft_net.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_clf_loader), eta_min=0,
                                                           last_epoch=-1)
    with torch.cuda.device(args.gpu_index):
        if len(args.ft_ids) > 1:
            args.dp = True
            ft_net = torch.nn.DataParallel(ft_net, device_ids=args.ft_ids)
        else:
            args.dp = False
        trainer = ASDTrainer(data_dir=args.data_dir,
                             id_fctor=args.ID_factor,
                             classifier=ft_net,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             args=args)
        trainer.train(train_clf_loader, contrain=args.contrain)


def test(args):
    net = SimCLRv2(num_class=args.num_classes)
    ft_net = SimCLRv2_ft(net, args.num_classes, use_arcface=args.use_arcface,
                         pretrain=args.pretrain, m=args.m, s=args.s, sub=args.sub_center)
    gpu_id = args.ft_ids[0]
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{gpu_id}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    with torch.cuda.device(args.gpu_index):
        opt = 'finetune' if args.pretrain else 'stgram-mfn'
        opt += '-contrain' if args.contrain else ''
        model_path = os.path.join(args.model_dir, args.version, opt,
                                  f'checkpoint_best.pth.tar')
        if len(args.ft_ids) > 1:
            args.dp = True
            net = torch.nn.DataParallel(net, device_ids=args.ft_ids)
        else:
            args.dp = False
        # load best model for test
        ft_net.load_state_dict(torch.load(model_path)['clf_state_dict'])
        trainer = ASDTrainer(data_dir=args.data_dir,
                             id_fctor=args.ID_factor,
                             classifier=net,
                             optimizer=None,
                             scheduler=None,
                             args=args)
        trainer.test(save=True)
        trainer.evaluator(save=True)
        for gmm_n in [64, 32, 16, 8, 4, 2, 1]:
            trainer.test(save=True, gmm_n=gmm_n)
            trainer.evaluator(save=True, gmm_n=gmm_n)


def main():
    # utils.replay_visdom(log_path='./log/')
    args = parser.parse_args()
    if args.seed: utils.setup_seed(args.seed)
    margin = args.ntxent_margin if args.pretrain_loss == 'ntxent' else args.supcon_margin
    version = "STgram-MFN"
    version += f'-Contrastive-pretrain-{args.pretrain_loss}(margin={margin},t={args.t},b={args.con_batch_size})-finetune' \
               if args.pretrain else ''
    version += f'-contrain-SupConLoss(margin={args.supcon_margin})' if args.contrain else ''
    version += f'-ArcFace(m={args.m},s={args.s},sub={args.sub_center})' if args.use_arcface else 'CELoss'
    args.version = version
    print(args.version)
    if args.pretrain: pretrain(args)
    finetune(args, epoch=100)
    test(args)


if __name__ == "__main__":
    main()
