from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from data_func.view_generator import ViewGenerator
import utils
# import WavAugment.augment as augment
import numpy as np
import joblib
import torchaudio
import librosa
import re
import os

class STgram_Contrastive_Dataset(Dataset):
    def __init__(self, dirs, ID_factor, sr,
                 win_length, hop_length, transform=None):
        self.factor = ID_factor
        filename_list_dict = {}
        for dir in dirs:
            filename_list = utils.get_filename_list(dir)
            for filename in filename_list:
                machine = filename.split('/')[-3]
                id_str = re.findall('id_[0-9][0-9]', filename)
                if machine == 'ToyCar' or machine == 'ToyConveyor':
                    id = int(id_str[0][-1]) - 1
                else:
                    id = int(id_str[0][-1])
                label = int(self.factor[machine] * 7 + id)
                if label in filename_list_dict.keys():
                    filename_list_dict[label].append(filename)
                else:
                    filename_list_dict[label] = [filename]
        self.filename_list_dict = filename_list_dict
        self.transform = transform
        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        # print(len(self.file_path_list))

    def __getitem__(self, item):
        wav_list, mel_list, label_list = [], [], []

        for index, label in enumerate(self.filename_list_dict.keys()):
            file_path = self.filename_list_dict[label][item]
            (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)
            x = x[:self.sr*10]  # (1, audio_length)
            x_wav = torch.from_numpy(x)
            x_mel = self.transform(x_wav).unsqueeze(0)
            if index == 0:
                wav_tensor = x_wav.unsqueeze(0)
                mel_tensor = x_mel
            else:
                wav_tensor = torch.cat((wav_tensor, x_wav.unsqueeze(0)), dim=0)
                mel_tensor = torch.cat((mel_tensor, x_mel), dim=0)

            label_list.append(label)
        return wav_tensor, mel_tensor, np.array(label_list)

    def __len__(self):
        for index, label in enumerate(self.filename_list_dict.keys()):
            if index == 0:
                min_len = len(self.filename_list_dict[label])
            else:
                if len(self.filename_list_dict[label]) < min_len:
                    min_len = len(self.filename_list_dict[label])
        return min_len

class Wav_Mel_ID_Dataset(Dataset):
    def __init__(self, dirs, ID_factor, sr,
                 win_length, hop_length, transform=None):
        self.filename_list = []
        for dir in dirs:
            self.filename_list.extend(utils.get_filename_list(dir))
        self.transform = transform
        self.factor = ID_factor
        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        # print(len(self.file_path_list))

    def __getitem__(self, item):
        file_path = self.filename_list[item]
        machine = file_path.split('/')[-3]
        id_str = re.findall('id_[0-9][0-9]', file_path)
        if machine == 'ToyCar' or machine == 'ToyConveyor':
            id = int(id_str[0][-1]) - 1
        else:
            id = int(id_str[0][-1])
        label = int(self.factor[machine] * 7 + id)
        (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)

        x = x[:self.sr*10]  # (1, audio_length)
        x_wav = torch.from_numpy(x)
        x_mel = self.transform(x_wav)
        return x_wav, x_mel, label

    def __len__(self):
        return len(self.filename_list)



class WavMelClassifierDataset:
    def __init__(self, dirs, sr, ID_factor, pattern="random"):
        self.dirs = dirs
        self.sr = sr
        self.factor = ID_factor
        self.pattern = pattern

    def get_dataset(self,
                    n_fft=1024,
                    n_mels=128,
                    win_length=1024,
                    hop_length=512,
                    power=2.0):
        if self.pattern == 'uniform':
            dataset = STgram_Contrastive_Dataset(self.dirs,
                                                 self.factor,
                                                 self.sr,
                                                 win_length,
                                                 hop_length,
                                                 transform=ViewGenerator(
                                                     self.sr,
                                                     n_fft=n_fft,
                                                     n_mels=n_mels,
                                                     win_length=win_length,
                                                     hop_length=hop_length,
                                                     power=power,
                                                 ))
            return dataset
        elif self.pattern == 'random':
            dataset = Wav_Mel_ID_Dataset(self.dirs,
                                         self.factor,
                                         self.sr,
                                         win_length,
                                         hop_length,
                                         transform=ViewGenerator(
                                             self.sr,
                                             n_fft=n_fft,
                                             n_mels=n_mels,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             power=power,
                                         ))
            return dataset
        else:
            assert "pattern only can be set as pretrain or finetune"



if __name__ == '__main__':

    pass

