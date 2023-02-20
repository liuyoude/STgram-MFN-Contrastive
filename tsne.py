from sklearn import datasets
from sklearn.manifold import TSNE
import os
import tqdm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import torch
import torch.nn.functional as F
import librosa

import utils
from simclrv2 import SimCLRv2, SimCLRv2_ft
from data_func.view_generator import ViewGenerator

# load yaml
param = utils.load_yaml(file_path='./config.yaml')
wav2mel = ViewGenerator(sr=param['sr'])

def transform(filename, device=torch.device('cuda:1')):
    sr = param['sr']
    (x, _) = librosa.core.load(filename, sr=sr, mono=True)
    x = x[:sr * 10]  # (1, audio_length)
    x = torch.from_numpy(x)
    x_mel = wav2mel(x)
    # x_mel = utils.normalize(x_mel, mean=self.args.mean, std=self.args.std)
    x = x.unsqueeze(0).unsqueeze(0).float().to(device)
    x_mel = x_mel.unsqueeze(0).unsqueeze(0).float().to(device)
    return x, x_mel

def get_latent_features(version, machine_type, epoch=120):
    n_mels = param['n_mels']
    frames = param['frames']
    data_dir = param['data_dir']
    model_dir = 'model_param'
    cuda = param['cuda']
    # load model
    model_path = os.path.join('./', model_dir, version, 'fine-tune', f'checkpoint_{epoch}.pth.tar')
    net = SimCLRv2()
    ft_net = SimCLRv2_ft(net, 41, arcface=False)
    ft_net = torch.nn.DataParallel(ft_net, device_ids=[4, 5])
    ft_net.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage.cuda() if cuda else 'cpu')['clf_state_dict'])
    model = ft_net.module
    if cuda:
        model.cuda(1)

    # get machine list
    target_dir = os.path.join(data_dir, machine_type)
    machine_id_list = utils.get_machine_id_list(target_dir)

    features = []
    id_labels = []
    anomaly_labels = []
    for id_idx, id_str in enumerate(machine_id_list):
        test_normal_files, test_anomaly_files = utils.create_wav_list(target_dir, id_str, dir_name='test')
        test_files = test_normal_files + test_anomaly_files
        id_labels += [id_idx for _ in range(len(test_files))]
        anomaly_labels += [0 for _ in range(len(test_normal_files))]
        anomaly_labels += [1 for _ in range(len(test_anomaly_files))]
        for file_idx, file_path in tqdm.tqdm(enumerate(test_files), total=len(test_files), desc=f'Getting test files features({id_str})'):
            x_wav, x_mel = transform(file_path)
            with torch.no_grad():
                model.eval()
                _, feature = model(x_wav, x_mel)
            feature = F.normalize(feature)
            features.append(feature.cpu())
    features = torch.cat(features, dim=0).numpy()
    return features, id_labels, anomaly_labels


def get_data(version, machine_type, epoch=120):
    label_desc = [
        # 'normal',
        'id_00', 'id_02', 'id_04', 'id_06'
        # '00_normal', '02_normal', '04_normal', '06_normal',
        # '00_anomaly', '02_anomaly', '04_anomaly', '06_anomaly',
        # 'anomaly',
    ]

    data, id_labels, anomaly_labels = get_latent_features(version, machine_type, epoch)
    data = data.reshape(data.shape[0], -1)
    return data, id_labels, anomaly_labels, label_desc


def plot_embedding(data, id_labels, anomaly_labels, label_desc, title, save_path, view='2D'):
    num_class = len(label_desc)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    shapes = ['o', 'x']
    fig = plt.figure(figsize=(12,6))
    if view == '3D':
        ax = Axes3D(fig)
        ax.scatter(data[:, 0],
                   data[:, 1],
                   data[:, 2],
                   c=plt.cm.rainbow(id_labels / num_class),
                   s=30,
                   alpha=0.8)
    else:
        for i in range(data.shape[0]):
            if shapes[anomaly_labels[i]] == 'o':
                plt.scatter(data[i, 0],
                            data[i, 1],
                            # str(label[i]),
                            facecolors=plt.cm.rainbow(id_labels[i] / num_class),
                            edgecolors='black',
                            s=40,
                            label=label_desc[id_labels[i]],
                            marker=shapes[anomaly_labels[i]],
                            alpha=0.8,
                            # fontdict={'weight': 'bold', 'size': 9},
                         )
            else:
                plt.scatter(data[i, 0],
                            data[i, 1],
                            # str(label[i]),
                            color=plt.cm.rainbow(id_labels[i] / num_class),
                            s=40,
                            label=label_desc[id_labels[i]],
                            marker=shapes[anomaly_labels[i]],
                            alpha=0.8,
                            # fontdict={'weight': 'bold', 'size': 9},
                            )

    i_list = list(range(num_class))
    patches = [mpatches.Patch(color=plt.cm.rainbow(i / num_class), label=f'{label_desc[i]}') for i in i_list]

    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    plt.legend(handles=patches, ncol=1, loc='upper right')
    plt.savefig(save_path, dpi=600)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    version = 'ID_Contrastive(Supcon)_STgram_MFN(pos_margin=False,t=0.1,lr=0.0005)_b=246_seed=526_ArcFace(m=1.0,s=30,sub=1)_300epochs_constrive=100epochs'
    machine_type = 'fan'
    view = '2D'
    save_path = os.path.join('./result', version, f't-SNE_{view}_{machine_type}_small.svg')

    data, id_labels, anomaly_labels, label_desc = get_data(version, machine_type, epoch='best')
    print(data.shape, len(id_labels), len(anomaly_labels))
    print('Computing t-SNE embedding')
    if view == '3D':
        tsne = TSNE(n_components=3, random_state=0, perplexity=30)
    else:
        tsne = TSNE(n_components=2, random_state=0, perplexity=20)
    result = tsne.fit_transform(data)
    plot_embedding(result, id_labels, anomaly_labels, label_desc, f't-SNE of {machine_type} latent features', save_path, view=view)