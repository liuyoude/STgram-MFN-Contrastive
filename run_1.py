import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
import yaml
import os

from data_func.dataset import *
from data_func.wave_split import data_split
from model import *
from trainer import *

from model import MobileFaceNet
from simclrv2 import SimCLRv2, SimCLRv2_ft

config_path = './config.yaml'
with open(config_path) as f:
    param = yaml.safe_load(f)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='STgram-MFN')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_false',
                    help='Disable CUDA')

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('-j', '--workers', default=param['workers'], type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=param['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=param['batch_size'], type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=param['lr'], type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=param['wd'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log-every-n-steps', default=20, type=int,
                    help='Log every n steps')

parser.add_argument('--model-dir', default=param['model_dir'], type=str, help='model saved dir')
parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result saved dir')
parser.add_argument('--data-dir', default=param['data_dir'], type=str, help='data dir')
parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str, help='processing data saved dir')
parser.add_argument('--data-type', default=param['data_type'], type=str, help='data type, dev_data or evl_data')
parser.add_argument('--split-flag', default=param['split_flag'], type=bool, help='pre-processing split or not')
parser.add_argument('--process-machines', default=param['process_machines'], type=list,
                    help='allowed processing machines')
parser.add_argument('--domain', default=param['domain'], type=str, help='process data domain, source or target')
parser.add_argument('--thread-num', default=param['thread_num'], type=int,
                    help='number of threading workers for data processing')

parser.add_argument('--sr', default=param['sr'], type=int, help='sample rate of wav files')
parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT param: n_fft')
parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT param: n_mels')
parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT param: hop_length')
parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT param: win_length')
parser.add_argument('--power', default=param['power'], type=float, help='STFT param: power')
parser.add_argument('--frames', default=param['frames'], type=int, help='split frames')
parser.add_argument('--skip-frames', default=param['skip_frames'], type=int, help='skip frames in spliting')

parser.add_argument('--pre-train', default=True, type=bool, help='pre train encoder with simclr or not')
parser.add_argument('--pre-train-epoch', default=10, type=int, help='epoch of pre train encoder')
parser.add_argument('--save-every-n-epochs', default=10, type=int, help='save encoder and decoder model every n epochs')
parser.add_argument('--early-stop', default=10, type=int, help='number of epochs for early stopping')

parser.add_argument('--con-lr', default=param['con_lr'], type=float)
parser.add_argument('--con_epochs', default=param['con_epochs'], type=int)
parser.add_argument('--con-batch-size', default=param['con_batch_size'], type=int)
parser.add_argument('--t', default=param['t'], type=float)

parser.add_argument('--version', default='Contrastive_STgram_MFN(t=0.05)_ArcFace(m=0.5,s=30)', type=str,
                    help='trail version')
parser.add_argument('--arcface', default=True, type=bool, help='using arcface or not')
parser.add_argument('--m', type=float, default=0.5, help='margin for arcface')
parser.add_argument('--s', type=float, default=30, help='scale for arcface')
parser.add_argument('--con-classes', default=param['con_classes'])
parser.add_argument('--con-file', default=param['con_file'])


def preprocess():
    args = parser.parse_args()
    root_folder = os.path.join(args.pre_data_dir, args.con_file)
    if not os.path.exists(root_folder):
        data_split(process_machines=args.process_machines,
                   data_dir=args.data_dir,
                   root_folder=root_folder,
                   ID_factor=param['ID_factor'])


def cs_test(args, cs=True, norm=True, epoch=80):
    con_model_path = os.path.join(args.model_dir, args.version, 'pre-train', f'checkpoint_{epoch:04d}.pth.tar')
    net = SimCLRv2(num_class=41, pretrained_weights=con_model_path)
    if args.arcface:
        arcface = ArcMarginProduct(128, 41, m=args.m, s=args.s)
    else:
        arcface = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda:7')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    with torch.cuda.device(args.gpu_index):
        model_path = os.path.join(args.model_dir, args.version, 'pre-train',
                                  f'checkpoint_0080.pth.tar')
        net = torch.nn.DataParallel(net, device_ids=[7, 3])
        if args.arcface:
            arcface = torch.nn.DataParallel(arcface, device_ids=[7, 3])
            arcface.load_state_dict(torch.load(model_path)['arcface_state_dict'])
        # ft_net.load_state_dict(torch.load(model_path)['clf_state_dict'])
        trainer = Contrastive_tester(data_dir=args.data_dir,
                                     id_fctor=param['ID_factor'],
                                     classifier=net,
                                     args=args)

        if cs or norm:
            cf_path = os.path.join('./data', f'{args.version}_center_feature.db')
            probs_path = os.path.join('./data', f'{args.version}_probs_mean_std.db')

            root_folder = os.path.join(args.pre_data_dir, f'313frames_train_path_list.db')
            clf_dataset = WavMelClassifierDataset(root_folder, args.sr, param['ID_factor'], pattern='fine_tune')
            train_clf_dataset = clf_dataset.get_dataset(n_mels=args.n_mels,
                                                        n_fft=args.n_fft,
                                                        hop_length=args.hop_length,
                                                        win_length=args.win_length,
                                                        power=args.power)
            train_clf_loader = torch.utils.data.DataLoader(
                train_clf_dataset, batch_size=32, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
            if not os.path.exists(cf_path) and cs:
                trainer.cal_class_center(train_clf_loader)
            if not os.path.exists(probs_path) and norm:
                trainer.save_mean_std(train_clf_loader)
        # max_recore = {}
        # w_dict = {}
        # for w in range(0, 102, 2):
        #     w /= 100
        #     print(f'\ncos_sim: probs = {w:.5f}:{(1 - w):.5f}')
        #     recore_dict = trainer.test(w, cs=cs, norm=norm, save=False)
        #     for key in recore_dict.keys():
        #         if key not in max_recore.keys():
        #             max_recore[key] = recore_dict[key]
        #             w_dict[key] = w
        #         else:
        #             if recore_dict[key] > max_recore[key]:
        #                 max_recore[key] = recore_dict[key]
        #                 w_dict[key] = w
        # print(w_dict)
        trainer.test(cs=cs, norm=norm, save=True)


def test(args):
    net = SimCLRv2()
    ft_net = SimCLRv2_ft(net, 41, args.arcface)
    if args.arcface:
        arcface = ArcMarginProduct(128, 41, m=args.m, s=args.s)
    else:
        arcface = None
    gpu_id = param['ft_ids'][0]
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{gpu_id}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    with torch.cuda.device(args.gpu_index):
        model_path = os.path.join(args.model_dir, args.version, 'fine-tune',
                                  f'checkpoint_best.pth.tar')
        ft_net = torch.nn.DataParallel(ft_net, device_ids=param['ft_ids'])
        if args.arcface:
            arcface = torch.nn.DataParallel(arcface, device_ids=param['ft_ids'])
            arcface.load_state_dict(torch.load(model_path)['arcface_state_dict'])
        # load best model for test

        ft_net.load_state_dict(torch.load(model_path)['clf_state_dict'])

        trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                       id_fctor=param['ID_factor'],
                                       classifier=ft_net,
                                       arcface=arcface,
                                       optimizer=None,
                                       scheduler=None,
                                       args=args)
        trainer.test(save=True)


def main(args):
    simclr_net = SimCLRv2(num_class=41)

    gpu_id = param['gpu_id']
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{gpu_id}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    root_folder = os.path.join(args.pre_data_dir, args.con_file)
    con_dataset = WavMelClassifierDataset(root_folder, args.sr, param['ID_factor'])
    train_clf_dataset = con_dataset.get_dataset(n_mels=args.n_mels,
                                                n_fft=args.n_fft,
                                                hop_length=args.hop_length,
                                                win_length=args.win_length,
                                                power=args.power)
    train_clf_loader = torch.utils.data.DataLoader(
        train_clf_dataset, batch_size=args.con_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    optimizer = torch.optim.Adam(simclr_net.parameters(), lr=args.con_lr)
    # optimizer = torch.optim.Adam([
    #     {'params':classfier.parameters(), 'lr':args.lr},
    #     {'params':FC_adacos.parameters(), 'lr':args.lr}
    # ])
    #
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_clf_loader), eta_min=0,
    #                                                        last_epoch=-1)
    #
    with torch.cuda.device(args.gpu_index):
        simclr_net = torch.nn.DataParallel(simclr_net, device_ids=param['device_ids'])
        trainer = CLR_trainer(data_dir=args.data_dir,
                              id_fctor=param['ID_factor'],
                              net=simclr_net,
                              optimizer=optimizer,
                              scheduler=None,
                              args=args)
        trainer.train(train_clf_loader)


def fine_tune(args, pretrain_path, epoch=20):
    con_model_path = os.path.join(args.model_dir, pretrain_path, 'pre-train', f'checkpoint_{epoch:04d}.pth.tar')
    net = SimCLRv2(pretrained_weights=con_model_path)
    ft_net = SimCLRv2_ft(net, 41, args.arcface)
    if args.arcface:
        arcface = ArcMarginProduct(128, 41, m=args.m, s=args.s)
    else:
        arcface = None
    gpu_id = param['ft_ids'][0]
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{gpu_id}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    root_folder = os.path.join(args.pre_data_dir, f'313frames_train_path_list.db')
    clf_dataset = WavMelClassifierDataset(root_folder, args.sr, param['ID_factor'], pattern='fine_tune')
    train_clf_dataset = clf_dataset.get_dataset(n_mels=args.n_mels,
                                                n_fft=args.n_fft,
                                                hop_length=args.hop_length,
                                                win_length=args.win_length,
                                                power=args.power)
    train_clf_loader = torch.utils.data.DataLoader(
        train_clf_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    # optimizer = torch.optim.Adam(ft_net.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam([
        {'params': ft_net.encoder.parameters(), 'lr': args.lr},
        {'params': ft_net.projector.parameters(), 'lr': args.lr},
        {'params': ft_net.linear.parameters(), 'lr': 1e-4}
    ])
    #
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_clf_loader), eta_min=0,
                                                           last_epoch=-1)
    #
    with torch.cuda.device(args.gpu_index):
        ft_net = torch.nn.DataParallel(ft_net, device_ids=param['ft_ids'])
        if args.arcface:
            arcface = torch.nn.DataParallel(arcface, device_ids=param['ft_ids'])
        trainer = wave_Mel_MFN_trainer(data_dir=args.data_dir,
                                       id_fctor=param['ID_factor'],
                                       classifier=ft_net,
                                       arcface=arcface,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       args=args)
        trainer.train(train_clf_loader)
        # load best model for test
        # model_path = os.path.join(args.model_dir, args.version, 'fine-tune',
        #                           f'checkpoint_best.pth.tar')
        # ft_net.load_state_dict(torch.load(model_path)['clf_state_dict'])
        # trainer.classifier = ft_net
        # trainer.test()


def reload_visdom(flag=False):
    if flag:
        writer = Visdom(env='main')
        utils.replay_visdom(writer, log_path='./log/')


if __name__ == "__main__":
    reload_visdom(flag=False)

    args = parser.parse_args()
    args.t = 0.005
    ver = f'ID_Contrastive_STgram_MFN(t={args.t},lr={args.con_lr})_b={args.con_batch_size}'
    args.version = ver
    print(args.version)
    preprocess()
    # main(args)

    args.arcface = False
    args.m = 0.5
    args.s = 30
    if args.arcface:
        args.version = ver + f'_ArcFace(m={args.m},s={args.s})'

    print(args.version)
    pretrain_path = ver
    cs_test(args, cs=True, norm=False, epoch=80)

    # =========================
    #  cos_sim test
    # =========================
    # args = parser.parse_args()
    # args.arcface = False
    # args.version = f'Contrastive_STgram_MFN(t=0.05)'
    # print(args.version)
    # cs_test(args, cs=True, norm=True)
