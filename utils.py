import os
import shutil
import sys

import torch
import yaml
import csv
import glob
import itertools
import re
import numpy as np
import librosa
import torch
import random


def load_yaml(file_path='./config.yaml'):
    with open(file_path) as f:
        params = yaml.safe_load(f)
    return params


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# file process
def select_dirs(data_dir, data_type='dev_data'):
    base_path = os.path.join(data_dir, data_type)
    dir_path = os.path.abspath(f'{base_path}/*')
    dirs = glob.glob(dir_path)
    return dirs


def replay_visdom(log_path):
    from visdom import Visdom
    writer = Visdom(env='main')
    file_path = os.path.abspath(f'{log_path}/*')
    files = glob.glob(file_path)
    for file in files:
        writer.replay_log(file)


def create_file_list(target_dir,
                     dir_name='train',
                     ext='wav'):
    list_path = os.path.abspath(f'{target_dir}/{dir_name}/*.{ext}')
    files = sorted(glob.glob(list_path))
    return files


def create_wav_list(target_dir,
                    id_name,
                    dir_name='test',
                    prefix_normal='normal',
                    prefix_anomaly='anomaly',
                    ext='wav'):
    normal_files_path = f'{target_dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))

    if dir_name == 'test':
        anomaly_files_path = f'{target_dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}'
        anomaly_files = sorted(glob.glob(anomaly_files_path))
        return normal_files, anomaly_files
    return normal_files


# get test machine id list
def get_machine_id_list(target_dir, ext='wav'):
    dir_path = os.path.abspath(f'{target_dir}/*.{ext}')
    files_path = sorted(glob.glob(dir_path))
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in files_path])
    )))
    return machine_id_list


def get_filename_list(dir_path, pattern='*', ext='*'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :param pattern: filename pattern for searching
    :return: files path list
    """
    filename_list = []
    for root, _, _ in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return filename_list


# def file_to_wav_vector(file_name,
#                        n_fft=1024,
#                        hop_length=512,
#                        frames=5,
#                        id_flag=False):
#     y, sr = librosa.load(file_name, sr=None)
#     wav_length = (frames-1) * hop_length + n_fft
#     # zero_padding = np.zeros(wav_length // 2)
#     # y = np.concatenate((y, zero_padding), axis=0)
#
#     wav_vector = np.zeros(((y.shape[0]-wav_length)//hop_length, wav_length))
#     for i in range(wav_vector.shape[0]):
#         wav_vector[i] = y[i*hop_length: i*hop_length+wav_length]
#
#     if id_flag:
#         id_str = re.findall('section_[0-9][0-9]', file_name)
#         id = int(id_str[0][-1])
#         id_vector = np.ones((wav_vector.shape[0], 1)) * id
#         return wav_vector, id_vector
#     return wav_vector

def file_to_wav_vector(file_name,
                       win_length=1024,
                       hop_length=512,
                       frames=5,
                       skip_frames=1):
    y, sr = librosa.load(file_name, sr=None)
    wav_length = (frames - 2 - 1) * hop_length + win_length
    skip_length = (skip_frames - 1) * hop_length
    wav_vector = np.zeros(((y.shape[0] - wav_length) // skip_length, wav_length))
    for i in range((y.shape[0] - wav_length) // skip_length):
        wav_vector[i] = y[i * skip_length: i * skip_length + wav_length]
    return wav_vector


# getting target dir file list and label list
def create_test_file_list(target_dir,
                          id_name,
                          dir_name='test',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


def create_train_file_list(target_dir,
                           id_name,
                           ext='wav'):
    files_path = f'{target_dir}/normal_{id_name}*.{ext}'
    files = sorted(glob.glob(files_path))
    return files


# make log mel spectrogram for each file
def file_to_log_mel_spectrogram(file_name,
                                n_mels=64,
                                n_fft=1024,
                                hop_length=512,
                                power=2.0,
                                p_flag=False):
    y, sr = librosa.load(file_name, sr=None)
    S = librosa.stft(y,
                     n_fft=n_fft,
                     hop_length=hop_length)
    p = np.angle(S)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    if p_flag:
        return log_mel_spectrogram, p
    return log_mel_spectrogram


def create_eval_file_list(target_dir,
                          id_name,
                          ext='wav'):
    files_path = f'{target_dir}/{id_name}*.{ext}'
    files = sorted(glob.glob(files_path))
    return files


# log mel spectrogram to vector
def log_mel_spect_to_vector(file_name,
                            skip_frames=1,
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0,
                            id_flag=False):
    """
    log mel spectrogram to vector
    :param frames: frames number concat as input
    :return: [vector_size, frames * n_mels]
    """
    log_mel_spect = file_to_log_mel_spectrogram(file_name,
                                                n_mels=n_mels,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
    dims = n_mels * frames
    vector_size = (log_mel_spect.shape[1] - frames) // skip_frames + 1
    if vector_size < 1:
        print('Warning: frames is too large!')
        return np.empty((0, dims))
    vector = np.zeros((vector_size, frames, n_mels))
    for n in range(vector_size):
        vector[n, :, :] = log_mel_spect[:, n * skip_frames: n * skip_frames + frames].T
    # for t in range(frames):
    #     vector[:, t * n_mels: (t + 1) * n_mels] = log_mel_spect[:, t: t + vector_size].T
    # vector = vector.reshape(vector_size, -1)
    if id_flag:
        id_str = re.findall('id_[0-9][0-9]', file_name)
        id = int(id_str[0][-1]) // 2
        id_vector = np.ones((vector_size, 1)) * id
        return vector, id_vector
    return vector


def calculate_gwrp(errors, decay):
    errors = sorted(errors, reverse=True)
    gwrp_w = decay ** np.arange(len(errors))
    # gwrp_w[gwrp_w < 0.1] = 0.1
    sum_gwrp_w = np.sum(gwrp_w)
    errors = errors * gwrp_w
    errors = np.sum(errors)
    score = errors / sum_gwrp_w
    return score


# calculate anomaly score
def calculate_anomaly_score(data,
                            predict_data,
                            pool_type='mean',
                            decay=1,
                            error_mode='square'):
    bs = data.shape[0]
    data = data.reshape(bs, -1)
    predict_data = predict_data.reshape(bs, -1)
    # mean score of n_mels for every frame
    if error_mode == 'square':
        errors = np.mean(np.square(data - predict_data), axis=1)
    else:
        errors = np.mean(np.abs(data - predict_data), axis=1)

    if pool_type == 'mean':
        score = np.mean(errors)
    elif pool_type == 'max':
        score = np.max(errors)
    elif pool_type == 'gwrp':
        score = calculate_gwrp(errors, decay)
    else:
        raise Exception(f'the pooling type is {pool_type}, mismatch with mean, max, max_frames_mean, gwrp, and gt_mean')

    return score


def cos_sim(a, b):
    v_a = np.mat(a)
    v_b = np.mat(b)
    num = float(v_a * v_b.T)
    denorm = np.linalg.norm(v_a) * np.linalg.norm(v_b)
    cos = num / denorm
    sim = 0.5 + 0.5 * cos
    return sim


def get_label(filename, factors):
    machine_type = filename.split('/')[-3]
    id_str = re.findall('id_[0-9][0-9]', filename)[0]
    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
        id = int(id_str[-1]) - 1
    else:
        id = int(id_str[-1])
    label = int(factors[machine_type] * 7 + id)
    return label


def model_ensemble(model_path, path_list, model_keys):
    ens_state_dict = {}
    for model_key in model_keys:
        sum = 0
        for idx, path in enumerate(path_list):
            model_state_dict = torch.load(path, map_location=torch.device('cpu'))[model_key]
            if idx == 0:
                ens_state_dict[model_key] = model_state_dict
            else:
                for key, value in model_state_dict.items():
                    ens_state_dict[model_key][key] += model_state_dict[key]
            sum += 1
        for key, value in ens_state_dict[model_key].items():
            ens_state_dict[model_key][key] = ens_state_dict[model_key][key] / sum
    torch.save(ens_state_dict, model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # version = 'ID_Contrastive_STgram_MFN(pos_margin=False,t=0.1,lr=0.0005)_b=6_ArcFace(m=1.0,s=30,sub=32)_300epochs_constrive=100epochs'
    version = 'ID_Contrastive_STgram_MFN(pos_margin=False,t=0.1,lr=0.0005)_b=6_ArcFace(m=1.0,s=30,sub=1)_300epochs_constrive=100epochs'
    path1 = f'./model_param/{version}/fine-tune/checkpoint_best.pth.tar'
    path_list = [path1]
    # for num in range(280, 300):
    for num in range(250, 291, 10):
        path = f'./model_param/{version}/fine-tune/checkpoint_0{num}.pth.tar'
        path_list.append(path)

    model_path = f'./model_param/{version}/fine-tune/checkpoint_ensemble-{len(path_list)}.pth.tar'
    model_ensemble(model_path, path_list, model_keys=['clf_state_dict', 'arcface_state_dict'])

    # path = './model_param/ID_Contrastive_STgram_MFN(pos_margin=0.2,t=0.01,lr=0.001)_b=6_ArcFace(m=1.0,s=30)/fine-tune/checkpoint_best.pth.tar'
    # state_dict = torch.load(path, map_location=torch.device('cpu'))['clf_state_dict']
    # for key, value in state_dict.items():
    #     print(key)