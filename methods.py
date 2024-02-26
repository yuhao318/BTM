import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, smooth_head, smooth_tail, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x

class ReweightedGMLLogAfterMean(nn.Module):
    '''
    This is the re-weighted generalized mean loss, which aims to maximize the generalized mean.

    Compared to ReweightedGML, here we put the negative log after we compute the per-class mean
    '''

    def __init__(self, num_class_list, p):
        super().__init__()
        # cfg = para_dict['cfg']
        # self.p = cfg.LOSS.REWEIGHTEDGMLLOGAFTERMEAN.P
        self.p = p
        print('p in Rweighted GML Log After Mean: ', self.p)
        self.device = "cuda:0"
        # self.device = para_dict['device']


        # self.weight = torch.Tensor(para_dict['num_class_list']).to(para_dict['device'])
        self.weight = torch.Tensor(num_class_list).to(self.device)

        print('Using re-weighted GML loss (Log After Mean)..')

    def forward(self, output, target):
        assert len(target.size()) == 1 # Target should be of 1-Dim

        max_logit = torch.max(output, dim=1, keepdim=True)[0] # of shape N x 1
        max_logit = max_logit.detach()
        logits = output - max_logit
        exp_logits = torch.exp(logits) * self.weight.view(-1, self.weight.shape[0])
        prob = torch.clamp(exp_logits / exp_logits.sum(1, keepdim=True), min=1e-5, max=1.)

        num_images, num_classes = prob.size()

        ground_class_prob = torch.gather(prob, dim=1, index=target.view(-1, 1))
        ground_class_prob = ground_class_prob.repeat((1, num_classes))

        mask = torch.zeros((num_images, num_classes), dtype=torch.int64, device=self.device)
        mask[range(num_images), target] = 1
        num_images_per_class = torch.sum(mask, dim=0)
        exist_class_mask = torch.zeros((num_classes,), dtype=torch.int64, device=self.device)
        exist_class_mask[num_images_per_class != 0] = 1

        num_images_per_class[num_images_per_class == 0] = 1 # avoid the dividing by zero exception

        mean_prob_classes = torch.sum(ground_class_prob * mask, dim=0) / num_images_per_class # of shape (C,)
        mean_prob_classes[exist_class_mask == 1] = -torch.log(mean_prob_classes[exist_class_mask == 1])

        mean_prob_sum = torch.sum(torch.pow(mean_prob_classes[exist_class_mask == 1], self.p)) / torch.sum(exist_class_mask)

        loss = torch.pow(mean_prob_sum, 1.0 / self.p)

        return loss
