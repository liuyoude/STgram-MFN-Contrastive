import logging
import os
import sys
import sklearn
import numpy as np
import time
import re
import joblib

import torch
import librosa
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import spafe.fbanks.gammatone_fbanks as gf
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
from tqdm import tqdm
from data_func.view_generator import ViewGenerator
from utils import accuracy, save_checkpoint, get_machine_id_list, create_test_file_list, log_mel_spect_to_vector, \
    calculate_anomaly_score, file_to_wav_vector
from NT_Xent import NT_Xent, SupconLoss

import utils


# torch.manual_seed(666)


class ASDTrainer(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.data_dir = kwargs['data_dir']
        self.id_factor = kwargs['id_fctor']
        self.machine_type = os.path.split(self.data_dir)[1]
        self.classifier = kwargs['classifier'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        visdom_folder = f'./log/'
        os.makedirs(visdom_folder, exist_ok=True)
        visdom_path = os.path.join(visdom_folder, f'{self.args.version}_visdom_ft.log')
        self.writer = Visdom(env=self.args.version, log_to_filename=visdom_path)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.supcon_criterion = SupconLoss(temperature=self.args.t,
                                           m=self.args.supcon_margin)
        self.wav2mel = ViewGenerator(sr=self.args.sr)
        self.csv_lines = []

    def train(self, train_loader, contrain=False):
        # self.test()
        n_iter = 0
        # create model dir for saving
        opt = 'finetune' if self.args.pretrain else 'stgram-mfn'
        opt += '-contrain' if self.args.contrain else ''
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, opt), exist_ok=True)
        print(f"Start finetune training for {self.args.epochs} epochs.")
        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        for epoch in range(self.args.epochs):
            pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
            for waveform, melspec, labels in pbar:
                waveform = waveform.float().to(self.args.device)
                melspec = melspec.float().to(self.args.device)
                labels = labels.long().reshape(-1, 1).to(self.args.device)
                self.classifier.train()
                predict_ids, hs, zs = self.classifier(waveform, melspec, labels.reshape(-1))
                loss_clf = self.criterion(predict_ids, labels.reshape(-1))
                loss_con = self.supcon_criterion(zs, labels) if contrain else torch.tensor(0).to(loss_clf.device)
                loss = loss_clf + self.args.lamda * loss_con
                pbar.set_description(f'Epoch:{epoch}'
                                     f'Loss:{loss.item():.5f}\tLclf:{loss_clf.item():.5f}\tLcon:{loss_con.item():.5f}')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.line([[loss.item(), loss_clf.item(), loss_con.item()]], [n_iter],
                                     win='Loss',
                                     update='append',
                                     opts=dict(
                                         title='Loss',
                                         legend=['loss', 'loss_clf', 'loss_con']
                                     ))
                    if self.scheduler is not None:
                        self.writer.line([self.scheduler.get_last_lr()[0]], [n_iter],
                                         win='Classifier LR',
                                         update='append',
                                         opts=dict(
                                             title='AE Learning Rate',
                                             legend=['lr']
                                         ))

                n_iter += 1

            if self.scheduler is not None and epoch >= 20:
                self.scheduler.step()
            print(f'Epoch:{epoch}'
                  f'Loss:{loss.item():.5f}\tLclf:{loss_clf.item():.5f}\tLcon:{loss_con.item():.5f}')
            if epoch % 2 == 0:
                # save model checkpoints
                auc, pauc = self.test()
                self.writer.line([[auc, pauc]], [epoch], win=self.machine_type,
                                 update='append',
                                 opts=dict(
                                     title=self.machine_type,
                                     legend=['AUC_clf', 'pAUC_clf']
                                 ))
                print(f'{self.machine_type}\t[{epoch}/{self.args.epochs}]\tAUC: {auc:3.3f}\tpAUC: {pauc:3.3f}')
                if (auc + pauc) > best_auc:
                    no_better = 0
                    best_auc = pauc + auc
                    p = pauc
                    a = auc
                    e = epoch
                    checkpoint_name = 'checkpoint_best.pth.tar'
                    save_checkpoint({
                        'epoch': epoch,
                        'clf_state_dict': self.classifier.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False,
                        filename=os.path.join(self.args.model_dir, self.args.version, opt, checkpoint_name))
                else:
                    no_better += 1
                # early stop
                # if no_better > 10:
                #     break

            if self.args.epochs - epoch <= 20:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch)
                save_checkpoint({
                    'epoch': epoch,
                    'clf_state_dict': self.classifier.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False,
                    filename=os.path.join(self.args.model_dir, self.args.version, opt, checkpoint_name))
        print(f'Traing {self.machine_type} completed!\tBest Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')

    def test(self, save=False, gmm_n=False):
        """
            gmm_n if set as number, using GMM estimator (n_components of GMM = gmm_n)
            if gmm_n = sub_center(arcface), using weight vector of arcface as the mean vector of GMM
        """
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        if gmm_n:
            result_dir = os.path.join(self.args.result_dir, self.args.version, f'GMM-{gmm_n}')
        os.makedirs(result_dir, exist_ok=True)
        self.classifier.eval()
        classifier = self.classifier.module if self.args.dp else self.classifier
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(self.args.valid_dirs)):
            time.sleep(1)
            machine_type = os.path.split(target_dir)[1]
            # result csv
            csv_lines.append([machine_type])
            csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                if gmm_n:
                    train_files = utils.create_train_file_list(target_dir, id_str)
                    features = self.get_ID_latent_features(train_files)
                    label = utils.get_label(train_files[0], self.id_factor)
                    means_init = classifier.arcface.weight[label * gmm_n: (label + 1) * gmm_n, :].detach().cpu().numpy() \
                        if self.args.use_arcface and (gmm_n == self.args.sub_center) else None
                    gmm = self.fit_GMM(features, n_components=gmm_n, means_init=means_init)
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    with torch.no_grad():
                        predict_ids, feature, _ = classifier(x_wav, x_mel, label)
                    if gmm_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(feature))
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            print(machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)
            csv_lines.append(['Average'] + list(averaged_performance))
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        csv_lines.append(['Total Average', avg_auc, avg_pauc])
        print('Total average:', avg_auc, avg_pauc)
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, csv_lines)
        return avg_auc, avg_pauc

    def evaluator(self, save=True, gmm_n=False):
        dirs = utils.select_dirs(self.data_dir, data_type='eval_dataset')
        # result_dir = os.path.join(self.args.result_dir, self.args.version)
        result_dir = os.path.join('./dcase2020_task2_evaluator-master/teams', self.args.version)
        if gmm_n:
            result_dir = os.path.join('./dcase2020_task2_evaluator-master/teams', self.args.version + f'-gmm-{gmm_n}')
        os.makedirs(result_dir, exist_ok=True)

        self.classifier.eval()
        classifier = self.classifier.module if self.args.dp else self.classifier
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(self.args.test_dirs)):
            time.sleep(1)
            machine_type = os.path.split(target_dir)[1]
            # get machine list
            machine_id_list = get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                test_files = utils.create_eval_file_list(target_dir, id_str)
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]

                if gmm_n:
                    train_files = utils.create_train_file_list(target_dir, id_str)
                    features = self.get_ID_latent_features(train_files)
                    label = utils.get_label(train_files[0], self.id_factor)
                    means_init = classifier.arcface.weight[label * gmm_n: (label + 1) * gmm_n, :].detach().cpu().numpy() \
                        if self.args.use_arcface and (gmm_n == self.args.sub_center) else None
                    # means_init = None
                    gmm = self.fit_GMM(features, n_components=gmm_n, means_init=means_init)
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    with torch.no_grad():
                        predict_ids, feature, _ = classifier(x_wav, x_mel, label)
                    if gmm_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(feature))
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)

    def get_ID_latent_features(self, train_files):
        pbar = tqdm(enumerate(train_files), total=len(train_files))
        self.classifier.eval()
        classifier = self.classifier.module if self.args.dp else self.classifier
        features = []
        for file_idx, file_path in pbar:
            x_wav, x_mel, label = self.transform(file_path)
            with torch.no_grad():
                _, feature, _ = classifier(x_wav, x_mel)
            if file_idx == 0:
                features = feature.cpu()
            else:
                features = torch.cat((features.cpu(), feature.cpu()), dim=0)
        if self.args.use_arcface: features = F.normalize(features)
        return features.numpy()

    def fit_GMM(self, data, n_components, means_init=None):
        print('=' * 40)
        print('Fit GMM in train data for test...')
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              means_init=means_init, reg_covar=1e-3, verbose=2)
        gmm.fit(data)
        print('Finish GMM fit.')
        print('=' * 40)
        return gmm

    def transform(self, filename):
        label = utils.get_label(filename, self.id_factor)
        (x, _) = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[:self.args.sr * 10]  # (1, audio_length)
        x = torch.from_numpy(x)
        x_mel = self.wav2mel(x)
        # x_mel = utils.normalize(x_mel, mean=self.args.mean, std=self.args.std)
        label = torch.from_numpy(np.array(label)).long().to(self.args.device)
        x = x.unsqueeze(0).float().to(self.args.device)
        x_mel = x_mel.unsqueeze(0).float().to(self.args.device)
        return x, x_mel, label


class CLRTrainer(object):
    """
        trainer for contrastive learning pretrain
    """

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.data_dir = kwargs['data_dir']
        self.id_factor = kwargs['id_fctor']
        self.machine_type = os.path.split(self.data_dir)[1]
        self.net = kwargs['net'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        visdom_folder = f'./log/'
        os.makedirs(visdom_folder, exist_ok=True)
        visdom_path = os.path.join(visdom_folder, f'{self.args.version}_visdom.log')
        self.writer = Visdom(env=self.args.version, log_to_filename=visdom_path)
        self.ntxent_criterion = NT_Xent(batch_size=self.args.con_batch_size,
                                        temperature=self.args.t,
                                        num_class=self.args.num_classes,
                                        m=self.args.ntxent_margin)
        self.supcon_criterion = SupconLoss(temperature=self.args.t,
                                           m=self.args.supcon_margin)
        self.csv_lines = []

    def train(self, train_loader):
        # self.eval()
        n_iter = 0
        # create model dir for saving
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, 'pretrain'), exist_ok=True)
        print(f"Start contrastive learning  pretraining for {self.args.con_epochs} epochs.")
        self.net.train()
        for epoch in range(self.args.con_epochs):
            pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
            for waveform, melspec, labels in pbar:
                b, n, l = waveform.shape
                _, _, f, t = melspec.shape
                waveform = waveform.float().reshape(b*n, l).to(self.args.device)
                melspec = melspec.float().reshape(b*n, f, t).to(self.args.device)
                labels = labels.long().squeeze().to(self.args.device)
                _, z = self.net(waveform, melspec, labels)
                loss = self.ntxent_criterion(z) if self.args.pretrain_loss == 'ntxent' else self.supcon_criterion(z, labels)
                pbar.set_description(f'Epoch:{epoch}'
                                     f'\tLclf:{loss.item():.5f}\t')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.line([loss.item()], [n_iter],
                                     win='Contrastive Loss',
                                     update='append',
                                     opts=dict(
                                         title='Contrastive Loss',
                                         legend=['loss']
                                     ))

                n_iter += 1

            if epoch >= 80 and epoch % 10 == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch)
                save_checkpoint({
                    'epoch': epoch,
                    'encoder': self.net.module.encoder.state_dict(),
                    'projector': self.net.module.projector.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False,
                    filename=os.path.join(self.args.model_dir, self.args.version, 'pretrain', checkpoint_name))
