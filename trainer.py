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
from NT_Xent import NT_Xent

import utils

# torch.manual_seed(666)


class wave_Mel_MFN_trainer(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.data_dir = kwargs['data_dir']
        self.id_factor = kwargs['id_fctor']
        self.machine_type = os.path.split(self.data_dir)[1]
        self.classifier = kwargs['classifier'].to(self.args.device)
        self.arcface = kwargs['arcface']
        if self.arcface is not None:
            self.arcface = self.arcface.to(self.args.device)
        # self.loss_layer = kwargs['loss_layer'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        visdom_folder = f'./log/'
        os.makedirs(visdom_folder, exist_ok=True)
        visdom_path = os.path.join(visdom_folder, f'{self.args.version}_visdom_ft.log')
        self.writer = Visdom(env=self.args.version, log_to_filename=visdom_path)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.recon_criterion = torch.nn.L1Loss().to(self.args.device)

        self.csv_lines = []


    def train(self, train_loader):
        # self.eval()

        # scaler = GradScaler(enabled=self.args.fp16_precision)

        n_iter = 0

        # create model dir for saving
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, 'fine-tune'), exist_ok=True)

        print(f"Start classifier training for {self.args.epochs} epochs.")
        print(f"Training with gpu: {self.args.disable_cuda}.")

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        for epoch_counter in range(self.args.epochs):
            pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
            for waveform, melspec, labels in pbar:
                waveform = waveform.float().unsqueeze(1).to(self.args.device)
                melspec = melspec.float().to(self.args.device)
                labels = labels.long().squeeze().to(self.args.device)
                self.classifier.train()
                predict_ids, _ = self.classifier(waveform, melspec)
                if self.arcface is not None:
                    self.arcface.train()
                    predict_ids = self.arcface(predict_ids, labels)
                loss_clf = self.criterion(predict_ids, labels)
                loss = loss_clf
                pbar.set_description(f'Epoch:{epoch_counter}'
                                     f'\tLclf:{loss_clf.item():.5f}\t')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.line([loss.item()], [n_iter],
                                     win='Classifier Loss',
                                     update='append',
                                     opts=dict(
                                         title='Classifier Loss',
                                         legend=['loss']
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

            if self.scheduler is not None and epoch_counter >= 20:
                self.scheduler.step()
            print(f"Epoch: {epoch_counter}\tLoss: {loss}")
            if epoch_counter % 2 == 0:
                # save model checkpoints
                auc, pauc = self.eval()
                self.writer.line([[auc, pauc]], [epoch_counter], win=self.machine_type,
                                 update='append',
                                 opts=dict(
                                     title=self.machine_type,
                                     legend=['AUC_clf', 'pAUC_clf']
                                 ))
                print(f'{self.machine_type}\t[{epoch_counter}/{self.args.epochs}]\tAUC: {auc:3.3f}\tpAUC: {pauc:3.3f}')
                if (auc + pauc) > best_auc:
                    no_better = 0
                    best_auc = pauc + auc
                    p = pauc
                    a = auc
                    e = epoch_counter
                    checkpoint_name = 'checkpoint_best.pth.tar'
                    if self.arcface is not None:
                        save_checkpoint({
                            'epoch': epoch_counter,
                            'clf_state_dict': self.classifier.state_dict(),
                            'arcface_state_dict': self.arcface.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, is_best=False,
                            filename=os.path.join(self.args.model_dir, self.args.version, 'fine-tune', checkpoint_name))
                    else:
                        save_checkpoint({
                            'epoch': epoch_counter,
                            'clf_state_dict': self.classifier.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, is_best=False,
                            filename=os.path.join(self.args.model_dir, self.args.version, 'fine-tune', checkpoint_name))
                else:
                    no_better += 1
                # if no_better > 10:
                #     break

            if epoch_counter > 100 and epoch_counter % 10 == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                if self.arcface is not None:
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'clf_state_dict': self.classifier.state_dict(),
                        'arcface_state_dict': self.arcface.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False,
                        filename=os.path.join(self.args.model_dir, self.args.version, 'fine-tune', checkpoint_name))
                else:
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'clf_state_dict': self.classifier.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False,
                        filename=os.path.join(self.args.model_dir, self.args.version, 'fine-tune', checkpoint_name))
        print(f'Traing {self.machine_type} completed!\tBest Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')


    def eval(self):
        sum_auc, sum_pauc, num, total_time = 0, 0, 0, 0
        #
        # sum_auc_r, sum_pauc_r = 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(dirs)):
            start = time.perf_counter()
            machine_type = os.path.split(target_dir)[1]
            if machine_type not in self.args.process_machines:
                continue
            num += 1
            # get machine list
            machine_id_list = get_machine_id_list(target_dir, dir_name='test')
            performance = []
            performance_recon = []
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                y_pred = [0. for _ in test_files]
                # y_pred_recon = [0. for _ in test_files]
                # print(111, len(test_files), target_dir)
                for file_idx, file_path in enumerate(test_files):
                    id_str = re.findall('id_[0-9][0-9]', file_path)
                    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                        id = int(id_str[0][-1]) - 1
                    else:
                        id = int(id_str[0][-1])
                    label = int(self.id_factor[machine_type] * 7 + id)
                    labels = torch.from_numpy(np.array(label)).long().to(self.args.device)
                    (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                    x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
                    x_wav = torch.from_numpy(x_wav)
                    x_wav = x_wav.float().to(self.args.device)

                    x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                    x_mel = torch.from_numpy(x_mel)
                    x_mel = ViewGenerator(self.args.sr,
                                          n_fft=self.args.n_fft,
                                          n_mels=self.args.n_mels,
                                          win_length=self.args.win_length,
                                          hop_length=self.args.hop_length,
                                          power=self.args.power,
                                          )(x_mel).unsqueeze(0).unsqueeze(0).to(self.args.device)

                    with torch.no_grad():
                        self.classifier.eval()
                        predict_ids, _ = self.classifier.module(x_wav, x_mel)
                        if self.arcface is not None:
                            self.arcface.eval()
                            predict_ids = predict_ids
                            labels = labels
                            predict_ids = self.arcface.module(predict_ids, labels)
                            predict_ids = predict_ids

                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = probs[label]

                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
            sum_auc += mean_auc
            sum_pauc += mean_p_auc

            time_nedded = time.perf_counter() - start
            total_time += time_nedded
            print(f'Test {machine_type} cost {time_nedded} secs')
        print(f'Total test time: {total_time} secs!')
        return sum_auc / num, sum_pauc / num


    def test(self, save=True):
        recore_dict = {}
        if not save:
            self.csv_lines = []

        sum_auc, sum_pauc, num = 0, 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(dirs)):
            time.sleep(1)
            machine_type = os.path.split(target_dir)[1]
            if machine_type not in self.args.process_machines:
                continue
            num += 1
            # result csv
            self.csv_lines.append([machine_type])
            self.csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = get_machine_id_list(target_dir, dir_name='test')
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'{machine_type}_anomaly_score_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                        id = int(id_str[-1]) - 1
                    else:
                        id = int(id_str[-1])
                    label = int(self.id_factor[machine_type] * 7 + id)
                    labels = torch.from_numpy(np.array(label)).long()
                    (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                    x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
                    x_wav = torch.from_numpy(x_wav)
                    x_wav = x_wav.float()

                    x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                    x_mel = torch.from_numpy(x_mel)
                    x_mel = ViewGenerator(self.args.sr,
                                          n_fft=self.args.n_fft,
                                          n_mels=self.args.n_mels,
                                          win_length=self.args.win_length,
                                          hop_length=self.args.hop_length,
                                          power=self.args.power,
                                          )(x_mel).unsqueeze(0).unsqueeze(0)

                    with torch.no_grad():
                        self.classifier.eval()
                        predict_ids, feature = self.classifier(x_wav, x_mel)
                        if self.arcface is not None:
                            self.arcface.eval()
                            predict_ids = predict_ids.repeat(2, 1)
                            labels = labels.repeat(2)
                            predict_ids = self.arcface(predict_ids, labels)
                            predict_ids = predict_ids[0:1, :]
                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()

                    y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                #
                self.csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            print(machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)
            recore_dict[machine_type] = mean_auc + mean_p_auc
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            self.csv_lines.append(['Average'] + list(averaged_performance))
        self.csv_lines.append(['Total Average', sum_auc / num, sum_pauc / num])
        print('Total average:', sum_auc / num, sum_pauc / num)
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, self.csv_lines)
        return recore_dict


    def TFE_visual(self, wav_path):
        (x, _) = librosa.core.load(wav_path, sr=self.args.sr, mono=True)
        x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x_wav)
        x_wav = x_wav.float()

        x_mel = x[:self.args.sr * 10]  # (1, audio_length)
        x_mel = torch.from_numpy(x_mel)
        x_mel = ViewGenerator(self.args.sr,
                              n_fft=self.args.n_fft,
                              n_mels=self.args.n_mels,
                              win_length=self.args.win_length,
                              hop_length=self.args.hop_length,
                              power=self.args.power,
                              )(x_mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.classifier.eval()
            wav_enco, mel = self.classifier(x_wav, x_mel)
        wav_enco = wav_enco.cpu().squeeze().flip(dims=(0,)).numpy()
        mel = mel.cpu().squeeze().flip(dims=(0,)).numpy()
        print(mel.shape)
        cmap = ['magma', 'inferno', 'plasma', 'hot']
        index = 1
        plt.imshow(mel, cmap=cmap[index])
        plt.axis('off')
        plt.show()
        plt.close()
        plt.imshow(wav_enco, cmap=cmap[index])
        plt.axis('off')
        plt.show()
        plt.close()


class CLR_trainer(object):

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
        self.criterion = NT_Xent(batch_size=self.args.con_batch_size,
                                 temperature=self.args.t,
                                 world_size=self.args.con_classes,
                                 m=self.args.pos_margin)

        self.csv_lines = []

    def train(self, train_loader):
        # self.eval()
        n_iter = 0
        # create model dir for saving
        os.makedirs(os.path.join(self.args.model_dir, self.args.version, 'pre-train'), exist_ok=True)

        print(f"Start classifier training for {self.args.con_epochs} epochs.")
        print(f"Training with gpu: {self.args.disable_cuda}.")

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        self.net.train()
        for epoch_counter in range(self.args.con_epochs):
            pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
            for waveform, melspec, labels in pbar:
                waveform = waveform.float().unsqueeze(2).to(self.args.device)
                melspec = melspec.float().to(self.args.device)

                _, z = self.net(waveform, melspec)

                loss = self.criterion(z)
                pbar.set_description(f'Epoch:{epoch_counter}'
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

            if epoch_counter % 5 == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                    'epoch': epoch_counter,
                    'encoder': self.net.module.encoder.state_dict(),
                    'projector': self.net.module.projector.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False,
                    filename=os.path.join(self.args.model_dir, self.args.version, 'pre-train', checkpoint_name))


class wave_Mel_MFN_tester(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.data_dir = kwargs['data_dir']
        self.id_factor = kwargs['id_fctor']
        self.machine_type = os.path.split(self.data_dir)[1]
        self.classifier = kwargs['classifier'].to(self.args.device)
        self.arcface = kwargs['arcface']
        if self.arcface is not None:
            self.arcface = self.arcface.to(self.args.device)
        self.writer = Visdom(env=self.args.version)

        self.csv_lines = []

    def save_mean_std(self, train_loader):
        cf_path = os.path.join('./data', f'{self.args.version}_center_feature.db')
        with open(cf_path, 'rb') as f:
            center_dict = joblib.load(f)

        os.makedirs(os.path.join(self.args.model_dir, self.args.version, 'Train'), exist_ok=True)
        probs_save_path = os.path.join('./data', f'{self.args.version}_probs_mean_std.db')
        cs_save_path = os.path.join('./data', f'{self.args.version}_cs_mean_std.db')
        probs_data = {}
        cs_data = {}
        pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
        for waveform, melspec, labels in pbar:
            waveform = waveform.float().unsqueeze(1).to(self.args.device)
            melspec = melspec.float().to(self.args.device)
            labels = labels.long().squeeze().to(self.args.device)
            self.classifier.eval()
            pridect_ids, features = self.classifier(waveform, melspec)
            if self.arcface is not None:
                self.arcface.eval()
                predict_ids = self.arcface(pridect_ids, labels)

            labels = labels.cpu().numpy().tolist()
            features = features.cpu().detach().numpy()
            for i in range(len(labels)):
                center_feature = center_dict[labels[i]]
                feature = features[i]

                probs = - torch.log_softmax(pridect_ids[i:i + 1], dim=1).mean(dim=0).squeeze().cpu().detach().numpy()
                cos_sim = utils.cos_sim(feature, center_feature)
                if labels[i] not in probs_data.keys():
                    probs_data[labels[i]] = []
                    cs_data[labels[i]] = []
                probs_data[labels[i]].append(probs[labels[i]])
                cs_data[labels[i]].append(cos_sim)

        probs_dict = {
            'mean': {},
            'std': {}
        }
        cs_dict = {
            'mean': {},
            'std': {}
        }
        for key in probs_data.keys():
            probs_dict['std'][key] = np.std(probs_data[key])
            probs_dict['mean'][key] = np.mean(probs_data[key])
            cs_dict['std'][key] = np.std(cs_data[key])
            cs_dict['mean'][key] = np.mean(cs_data[key])

        print(probs_dict['std'], cs_dict['std'])
        with open(probs_save_path, 'wb') as f:
            joblib.dump(probs_dict, f)
        with open(cs_save_path, 'wb') as f:
            joblib.dump(cs_dict, f)

    def cal_class_center(self, train_loader):
        # create model dir for saving
        save_path = os.path.join('./data', f'{self.args.version}_center_feature.db')
        data = {}
        class_num = {}
        pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
        for waveform, melspec, labels in pbar:
            waveform = waveform.float().unsqueeze(1).to(self.args.device)
            melspec = melspec.float().to(self.args.device)
            labels = labels.long().squeeze().numpy().tolist()
            self.classifier.eval()
            _, features = self.classifier(waveform, melspec)
            features = features.cpu().detach().numpy()
            for i in range(len(labels)):
                if labels[i] not in data.keys():
                    data[labels[i]] = features[i]
                    class_num[labels[i]] = 1
                else:
                    data[labels[i]] += features[i]
                    class_num[labels[i]] += 1

        for key in data.keys():
            data[key] /= class_num[key]
        with open(save_path, 'wb') as f:
            joblib.dump(data, f)

    def test(self, w, cs=False, norm=False, save=True):
        recore_dict = {}
        if not save:
            self.csv_lines = []
        if cs:
            save_path = os.path.join('./data', f'{self.args.version}_center_feature.db')
            with open(save_path, 'rb') as f:
                center_dict = joblib.load(f)
        if norm:
            probs_path = os.path.join('./data', f'{self.args.version}_probs_mean_std.db')
            cs_path = os.path.join('./data', f'{self.args.version}_cs_mean_std.db')
            with open(probs_path, 'rb') as f:
                probs_dict = joblib.load(f)
            with open(cs_path, 'rb') as f:
                cs_dict = joblib.load(f)

        sum_auc, sum_pauc, num = 0, 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        if cs:
            result_dir = os.path.join(result_dir, 'cos_sim+nega_prob')
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(dirs)):
            time.sleep(1)
            machine_type = os.path.split(target_dir)[1]
            if machine_type not in self.args.process_machines:
                continue
            num += 1
            # result csv
            self.csv_lines.append([machine_type])
            self.csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = get_machine_id_list(target_dir, dir_name='test')
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'{machine_type}_anomaly_score_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                        id = int(id_str[-1]) - 1
                    else:
                        id = int(id_str[-1])
                    label = int(self.id_factor[machine_type] * 7 + id)
                    labels = torch.from_numpy(np.array(label)).long()
                    (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                    x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
                    x_wav = torch.from_numpy(x_wav)
                    x_wav = x_wav.float()

                    x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                    x_mel = torch.from_numpy(x_mel)
                    x_mel = ViewGenerator(self.args.sr,
                                          n_fft=self.args.n_fft,
                                          n_mels=self.args.n_mels,
                                          win_length=self.args.win_length,
                                          hop_length=self.args.hop_length,
                                          power=self.args.power,
                                          )(x_mel).unsqueeze(0).unsqueeze(0)

                    with torch.no_grad():
                        self.classifier.eval()
                        predict_ids, feature = self.classifier(x_wav, x_mel)
                        if self.arcface is not None:
                            self.arcface.eval()
                            predict_ids = predict_ids.repeat(2, 1)
                            labels = labels.repeat(2)
                            predict_ids = self.arcface(predict_ids, labels)
                            predict_ids = predict_ids[0:1, :]
                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    if cs:
                        feature = feature.cpu().squeeze().numpy()
                        center_feature = center_dict[label]
                        if type(w) == dict:

                            if norm:
                                y_pred[file_idx] = -w[machine_type] * utils.cos_sim(feature, center_feature) + (
                                            probs[label] / probs_dict['std'][label])
                            else:
                                y_pred[file_idx] = - w[machine_type] * (
                                            utils.cos_sim(feature, center_feature) / cs_dict['std'][label]) + probs[
                                                       label]
                            # y_pred[file_idx] = - (utils.cos_sim(feature, center_feature) / cs_dict['std'][label])
                        else:
                            # print(utils.cos_sim(feature, center_feature), utils.cos_sim(feature, center_feature)/cs_dict['std'][label], probs[label] / probs_dict['std'][label])
                            if norm:
                                y_pred[file_idx] = -w * utils.cos_sim(feature, center_feature) + (
                                            probs[label] / probs_dict['std'][label])
                            else:
                                y_pred[file_idx] = - w * (
                                            utils.cos_sim(feature, center_feature) / cs_dict['std'][label]) + probs[
                                                       label]
                            # y_pred[file_idx] = - (utils.cos_sim(feature, center_feature) / cs_dict['std'][label])
                    else:
                        if norm:
                            y_pred[file_idx] = probs[label] / probs_dict['std'][label]
                        else:
                            y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                #
                self.csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            print(machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)
            recore_dict[machine_type] = mean_auc + mean_p_auc
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            self.csv_lines.append(['Average'] + list(averaged_performance))
        self.csv_lines.append(['Total Average', sum_auc / num, sum_pauc / num])
        print('Total average:', sum_auc / num, sum_pauc / num)
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, self.csv_lines)
        return recore_dict

    def get_latent_features(self, model, target_dir):
        # get machine list
        machine_id_list = utils.get_machine_id_list(target_dir)
        machine_type = os.path.split(target_dir)[1]
        if machine_type == 'ToyCar':
            label_desc = [
                'id_01', 'id_02', 'id_03', 'id_04'
            ]
        elif machine_type == 'ToyConveyor':
            label_desc = [
                'id_01', 'id_02', 'id_03'
            ]
        else:
            label_desc = [
                'id_00', 'id_02', 'id_04', 'id_06'
            ]

        features = []
        labels = []
        for id_idx, id_str in enumerate(machine_id_list):
            test_normal_files, test_anomaly_files = utils.create_wav_list(target_dir, id_str, dir_name='test')
            # train_normal_files = utils.create_wav_list(target_dir, id_str, dir_name='train'
            # test_normal_files = test_normal_files[:1]
            # test_anomaly_files = test_anomaly_files[:1
            for file_idx, file_path in tqdm(enumerate(test_normal_files), total=len(test_normal_files),
                                            desc=f'[{machine_type}]Getting test normal files features({id_str})'):
                if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                    id = int(id_str[-1]) - 1
                else:
                    id = int(id_str[-1]) // 2
                label = id

                (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
                x_wav = torch.from_numpy(x_wav)
                x_wav = x_wav.float()

                x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                x_mel = torch.from_numpy(x_mel)
                x_mel = ViewGenerator(self.args.sr,
                                      n_fft=self.args.n_fft,
                                      n_mels=self.args.n_mels,
                                      win_length=self.args.win_length,
                                      hop_length=self.args.hop_length,
                                      power=self.args.power,
                                      )(x_mel).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    model.eval()
                    _, out_feature = model(x_wav, x_mel)
                    # predict_ids, _ = self.classifier(data, None, None)
                out_feature = out_feature.cpu().view(out_feature.size(0), -1).numpy()
                if id_idx == 0 and file_idx == 0:
                    features = out_feature
                else:
                    features = np.concatenate((features, out_feature), axis=0)
                labels.append(label)
            for file_idx, file_path in tqdm(enumerate(test_anomaly_files), total=len(test_anomaly_files),
                                            desc=f'Getting test anomaly files features({id_str})'):
                if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                    id = int(id_str[-1]) - 1
                else:
                    id = int(id_str[-1]) // 2
                if machine_type == 'ToyConveyor':
                    label = id + 3
                else:
                    label = id + 4

                (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
                x_wav = torch.from_numpy(x_wav)
                x_wav = x_wav.float()

                x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                x_mel = torch.from_numpy(x_mel)
                x_mel = ViewGenerator(self.args.sr,
                                      n_fft=self.args.n_fft,
                                      n_mels=self.args.n_mels,
                                      win_length=self.args.win_length,
                                      hop_length=self.args.hop_length,
                                      power=self.args.power,
                                      )(x_mel).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    self.classifier.eval()
                    _, out_feature = model(x_wav, x_mel)
                out_feature = out_feature.cpu().view(out_feature.size(0), -1).numpy()
                features = np.concatenate((features, out_feature), axis=0)
                labels.append(label)
        return features, labels, label_desc

    def vis_tsne_signal(self, machine_type):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        save_path = os.path.join(result_dir, f'tsne_{machine_type}.png')
        target_dir = os.path.join(self.args.data_dir, machine_type)
        # machine_type = os.path.split(target_dir)[1]
        title = f'{machine_type}'
        data, label, label_desc = self.get_latent_features(self.classifier, target_dir)
        data = data.reshape(data.shape[0], -1)
        print(data.shape)
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        data = tsne.fit_transform(data)
        print(data.shape, len(label))
        num_class = len(label_desc)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        shapes = ['o', 'x']
        label_color = []
        if machine_type == 'ToyConveyor':
            normal_num = 3
        else:
            normal_num = 4
        for i in range(len(label)):
            if label[i] >= normal_num:
                label_color.append(label[i] - 4)
            else:
                label_color.append(label[i])
        label_shape = label

        for i in range(data.shape[0]):
            plt.scatter(data[i, 0],
                        data[i, 1],
                        # str(label[i]),
                        color=plt.cm.rainbow(label_color[i] / num_class),
                        s=60,
                        label=label_desc[label_color[i]],
                        marker=shapes[label_shape[i] // 4],
                        alpha=0.8
                        # fontdict={'weight': 'bold', 'size': 9},
                        )

        i_list = list(range(num_class))
        patches = [mpatches.Patch(color=plt.cm.rainbow(i / num_class), label=f'{label_desc[i]}') for i in i_list]

        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.legend(handles=patches, ncol=3, loc='best')
        plt.savefig(save_path, dpi=300)
        plt.show()

    def TFE_visual(self, wav_path):
        (x, _) = librosa.core.load(wav_path, sr=self.args.sr, mono=True)
        x_wav = x[None, None, :self.args.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x_wav)
        x_wav = x_wav.float()

        x_mel = x[:self.args.sr * 10]  # (1, audio_length)
        x_mel = torch.from_numpy(x_mel)
        x_mel = ViewGenerator(self.args.sr,
                              n_fft=self.args.n_fft,
                              n_mels=self.args.n_mels,
                              win_length=self.args.win_length,
                              hop_length=self.args.hop_length,
                              power=self.args.power,
                              )(x_mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.classifier.eval()
            wav_enco, mel = self.classifier(x_wav, x_mel)
        wav_enco = wav_enco.cpu().squeeze().flip(dims=(0,)).numpy()
        mel = mel.cpu().squeeze().flip(dims=(0,)).numpy()
        print(mel.shape)
        cmap = ['magma', 'inferno', 'plasma', 'hot']
        index = 1
        plt.imshow(mel, cmap=cmap[index])
        plt.axis('off')
        plt.show()
        plt.close()
        plt.imshow(wav_enco, cmap=cmap[index])
        plt.axis('off')
        plt.show()
        plt.close()

class Contrastive_tester(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.data_dir = kwargs['data_dir']
        self.id_factor = kwargs['id_fctor']
        self.machine_type = os.path.split(self.data_dir)[1]
        self.classifier = kwargs['classifier'].to(self.args.device)
        self.writer = Visdom(env=self.args.version)

        self.csv_lines = []

    def cal_class_center(self, train_loader):
        # create model dir for saving
        os.makedirs('./data', exist_ok=True)
        save_path = os.path.join('./data', f'{self.args.version}_center_feature.db')
        data = {}
        class_num = {}
        pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
        for waveform, melspec, labels in pbar:
            waveform = waveform.float().unsqueeze(1).unsqueeze(1).to(self.args.device)
            melspec = melspec.float().unsqueeze(1).to(self.args.device)
            labels = labels.long().squeeze().numpy().tolist()
            self.classifier.eval()
            _, features = self.classifier(waveform, melspec)
            features = features.cpu().detach().numpy()
            for i in range(len(labels)):
                if labels[i] not in data.keys():
                    data[labels[i]] = features[i]
                    class_num[labels[i]] = 1
                else:
                    data[labels[i]] += features[i]
                    class_num[labels[i]] += 1

        for key in data.keys():
            data[key] /= class_num[key]
        with open(save_path, 'wb') as f:
            joblib.dump(data, f)

    def test(self, cs=False, norm=False, save=True):
        recore_dict = {}
        if not save:
            self.csv_lines = []
        if cs:
            save_path = os.path.join('./data', f'{self.args.version}_center_feature.db')
            with open(save_path, 'rb') as f:
                center_dict = joblib.load(f)
        if norm:
            probs_path = os.path.join('./data', f'{self.args.version}_probs_mean_std.db')
            cs_path = os.path.join('./data', f'{self.args.version}_cs_mean_std.db')
            with open(probs_path, 'rb') as f:
                probs_dict = joblib.load(f)
            with open(cs_path, 'rb') as f:
                cs_dict = joblib.load(f)

        sum_auc, sum_pauc, num = 0, 0, 0
        dirs = utils.select_dirs(self.data_dir, data_type='')
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        if cs:
            result_dir = os.path.join(result_dir, 'cos_sim')
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(dirs)):
            time.sleep(1)
            machine_type = os.path.split(target_dir)[1]
            if machine_type not in self.args.process_machines:
                continue
            num += 1
            # result csv
            self.csv_lines.append([machine_type])
            self.csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = get_machine_id_list(target_dir, dir_name='test')
            for id_str in machine_id_list:
                test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'{machine_type}_anomaly_score_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                        id = int(id_str[-1]) - 1
                    else:
                        id = int(id_str[-1])
                    label = int(self.id_factor[machine_type] * 7 + id)
                    labels = torch.from_numpy(np.array(label)).long().to(self.args.device)
                    (x, _) = librosa.core.load(file_path, sr=self.args.sr, mono=True)

                    x_wav = x[None, None, None, :self.args.sr * 10]  # (1, audio_length)
                    x_wav = torch.from_numpy(x_wav)
                    x_wav = x_wav.float().to(self.args.device)

                    x_mel = x[:self.args.sr * 10]  # (1, audio_length)
                    x_mel = torch.from_numpy(x_mel)
                    x_mel = ViewGenerator(self.args.sr,
                                          n_fft=self.args.n_fft,
                                          n_mels=self.args.n_mels,
                                          win_length=self.args.win_length,
                                          hop_length=self.args.hop_length,
                                          power=self.args.power,
                                          )(x_mel).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.args.device)

                    with torch.no_grad():
                        self.classifier.eval()
                        predict_ids, feature = self.classifier.module(x_wav, x_mel)
                    feature = feature.cpu().squeeze().numpy()
                    center_feature = center_dict[label]
                    y_pred[file_idx] = - utils.cos_sim(feature, center_feature)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                #
                self.csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            print(machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)
            recore_dict[machine_type] = mean_auc + mean_p_auc
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            self.csv_lines.append(['Average'] + list(averaged_performance))
        self.csv_lines.append(['Total Average', sum_auc / num, sum_pauc / num])
        print('Total average:', sum_auc / num, sum_pauc / num)
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, self.csv_lines)
        return recore_dict