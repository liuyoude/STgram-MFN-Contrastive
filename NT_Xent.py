import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed as dist
import torch.nn.functional as F
from model import ArcMarginProduct
import math


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size, m=False):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask, self.not_posi_mask = self.mask_correlated_samples(batch_size, world_size)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.m = m
        if m:
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)
            # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m


    def mask_correlated_samples(self, batch_size, world_size):
        # world size is the number of different classes
        N = batch_size * world_size
        # mask = torch.ones((N, N), dtype=bool)
        mask = torch.ones((N, N), dtype=bool)


        for i in range(1, batch_size):
            for j in range(N - i * world_size):
                mask[i*world_size+j, j] = 0
                mask[j, i*world_size+j] = 0
        not_posi_mask = ~ mask
        mask = mask.fill_diagonal_(0)
        # mask = mask.fill_diagonal_(0)
        # for i in range(batch_size * world_size):
        #     mask[i, batch_size + i] = 0
        #     mask[batch_size + i, i] = 0
        # print(mask)
        return mask, not_posi_mask

    def forward(self, z):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

        N = self.batch_size * self.world_size

        # z = torch.cat((z_i, z_j), dim=0)
        # if self.world_size > 1:
        #     z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        #
        # for i, b in enumerate(range(1, self.batch_size)):
        #     if i == 0:
        #         sim_i_j = torch.diag(sim, b * self.world_size)
        #         sim_j_i = torch.diag(sim, - b * self.world_size)
        #     else:
        #         sim_i_j = torch.cat((sim_i_j, torch.diag(sim, b * self.world_size)))
        #         sim_j_i = torch.cat((sim_j_i, torch.diag(sim, - b * self.world_size)))
        # # sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        # # sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
        # print(sim_i_j.shape, sim_j_i.shape)
        # # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, self.batch_size-1)

        positive_samples = sim[self.not_posi_mask].reshape(N, self.batch_size-1)
        if self.m:
            cosine = positive_samples
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            positive_samples = phi

        negative_samples = sim[self.mask].reshape(N, N-self.batch_size)

        # TODO BCE -> CCE
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

if __name__ == '__main__':
    ntx = NT_Xent(32, 0.01, 6)
    z = torch.rand((32*6, 128))
    print(ntx(z))
    # print(ntx.mask_correlated_samples(5, 3))
