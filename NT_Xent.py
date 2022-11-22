import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed as dist
import torch.nn.functional as F
from model import ArcMarginProduct
import math


class NT_Xent(nn.Module):
    """
        supervised contrastive learning
        input need same samples from all different classes (i.e., batch size samples each class)
    """
    def __init__(self, batch_size, temperature, num_class, m=False):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.num_class = num_class
        self.eps = 1e-8

        self.neg_mask, self.pos_mask = self.mask_correlated_samples(batch_size, num_class)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.m = m
        # try to add margin for positive samples
        if m:
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)
            # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m

    def mask_correlated_samples(self, batch_size, num_class):
        N = batch_size * num_class
        neg_mask = torch.ones((N, N), dtype=bool)
        for i in range(1, batch_size):
            for j in range(N - i * num_class):
                neg_mask[i*num_class+j, j] = 0
                neg_mask[j, i*num_class+j] = 0
        pos_mask = ~ neg_mask
        neg_mask = neg_mask.fill_diagonal_(0)
        # mask = mask.fill_diagonal_(0)
        # for i in range(batch_size * world_size):
        #     mask[i, batch_size + i] = 0
        #     mask[batch_size + i, i] = 0
        # print(mask)
        return neg_mask, pos_mask

    def forward(self, z):
        N = self.batch_size * self.num_class
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        positive_samples = sim[self.pos_mask].reshape(N, self.batch_size-1)
        # try to add margin
        if self.m:
            cosine = positive_samples
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + self.eps)
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            positive_samples = phi
        negative_samples = sim[self.neg_mask].reshape(N, N-self.batch_size)
        # BCE loss
        labels = torch.zeros(N, N-1).to(positive_samples.device).long()
        labels[:, :self.batch_size-1] = 1
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature
        loss = self.criterion(logits, labels.float())
        # CCE loss
        # anchor_dot_contrast = torch.cat((positive_samples, negative_samples), dim=1)
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        #
        # logits_mask = torch.zeros(N, N - 1).to(positive_samples.device).bool()
        # logits_mask[:, :self.batch_size - 1] = 1
        #
        # exp_logits = torch.exp(logits)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        #
        # mean_log_prob_pos = (logits_mask * log_prob).sum(1) / logits_mask.sum(1)
        #
        # loss = -  mean_log_prob_pos
        # loss = loss.mean()
        return loss

class SupconLoss(nn.Module):
    def __init__(self, temperature, m=False):
        super(SupconLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.m = m
        if m:
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)
            # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m

    def forward(self, z, labels):
        """
        supervised contrastive learning
        random input samples
        """
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        #
        mask = torch.eq(labels, labels.T).to(sim.device)
        pos_mask = mask.fill_diagonal_(0)
        neg_mask = ~ pos_mask
        if self.m:
            cosine = sim.clone()
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + self.eps)
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            margin_sim = phi
            sim = pos_mask.float() * margin_sim + neg_mask.float() * sim
        # CCE loss
        anchor_dot_contrast = sim
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # delete no couple positive samples (only one class in batch)
        constrast_index = ~(pos_mask.sum(1, keepdim=True).long().repeat(1, mask.size(1)) == 0)
        logits_mask = pos_mask[constrast_index].reshape(-1, mask.size(1))
        # positive_mask = positive_mask.fill_diagonal_(0)[constrast_index].reshape(-1, mask.size(1))
        logits = logits[constrast_index].reshape(-1, mask.size(1)) / self.temperature
        # print(logits.shape)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log((exp_logits).sum(1, keepdim=True))
        mean_log_prob_pos = (logits_mask * log_prob).sum(1) / logits_mask.sum(1)
        loss = - mean_log_prob_pos[~(logits_mask.sum(1).long() == 0)]
        loss = loss.mean()
        return loss


if __name__ == '__main__':
    ntx = NT_Xent(32, 0.01, 6)
    z = torch.rand((4, 128))
    z = F.normalize(z)
    # print(ntx(z))
    # print(ntx.mask_correlated_samples(5, 3))
