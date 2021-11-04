import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dim_estimation.datasets.svoc import StylizedVoc


def get_dataloader(args):
    if args.dataset == 'svoc':
        dataset = StylizedVoc(args)
    else:
        print('dataset not available!')
        sys.exit()
    dataloader = DataLoader(dataset, args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            # pin_memory=True
                            )
    # pin_memory=True causes out-of-memory error on my machine
    return dataloader


class Distribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean = torch.chunk(parameters, 1, dim=1)
        self.deterministic = deterministic

    def sample(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = self.mean + self.std*torch.randn(self.mean.shape).to(device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5*torch.sum(torch.pow(self.mean, 2)
                        + self.var - 1.0 - self.logvar,
                        dim=[1,2,3])
            else:
                return 0.5*torch.sum(
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar + other.logvar,
                        dim=[1,2,3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0*np.pi)
        return 0.5*torch.sum(
                logtwopi+self.logvar+torch.pow(sample-self.mean, 2) / self.var,
                dim=[1,2,3])

    def mode(self):
        return self.mean


def dim_est(output_dict, factor_list, args):
    # grab flattened factors, examples
    # factors = data_out.labels["factor"]
    # za = data_out.labels["example1"].squeeze()
    # zb = data_out.labels["example2"].squeeze()

    # factors = np.random.choice(2, 21845) # shape=21845
    # za = np.random.rand(21845, 2048)
    # zb = np.random.rand(21845, 2048)

    za = np.concatenate(output_dict['example1'])
    zb = np.concatenate(output_dict['example2'])
    factors = np.concatenate(factor_list)


    za_by_factor = dict()
    zb_by_factor = dict()
    mean_by_factor = dict()
    score_by_factor = dict()
    individual_scores = dict()

    zall = np.concatenate([za, zb], 0)
    mean = np.mean(zall, 0, keepdims=True)

    # za_means = np.mean(za,axis=1)
    # zb_means = np.mean(zb,axis=1)
    # za_vars = np.mean((za - za_means[:, None]) * (za - za_means[:, None]), 1)
    # zb_vars = np.mean((za - zb_means[:, None]) * (za - zb_means[:, None]), 1)

    var = np.sum(np.mean((zall - mean) * (zall - mean), 0))
    for f in range(args.n_factors):
        if f != args.residual_index:
            # why is residual factor explicitly in data?
            # why not just append it and use all the data
            indices = np.where(factors == f)[0]
            za_by_factor[f] = za[indices]
            zb_by_factor[f] = zb[indices]
            mean_by_factor[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True) +
                                     np.mean(zb_by_factor[f], 0, keepdims=True))
            # score_by_factor[f] = np.sum(np.mean(np.abs((za_by_factor[f] - mean_by_factor[f])* (zb_by_factor[f] - mean_by_factor[f])), 0))
            # score_by_factor[f] = score_by_factor[f] / var
            # OG
            score_by_factor[f] = np.sum(np.mean((za_by_factor[f] - mean_by_factor[f]) *
                                                (zb_by_factor[f] - mean_by_factor[f]), 0))
            score_by_factor[f] = score_by_factor[f] / var

            idv = np.mean((za_by_factor[f] - mean_by_factor[f]) *
                          (zb_by_factor[f] - mean_by_factor[f]), 0) / var
            individual_scores[f] = idv
        #   new method
        #     score_by_factor[f] = np.abs(np.mean((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f]), 0))
        #     score_by_factor[f] = np.sum(score_by_factor[f])
        #     score_by_factor[f] = score_by_factor[f] / var

            # new with threshhold
            # sigmoid
            # score_by_factor[f] = sigmoid(score_by_factor[f])
            # score_by_factor[f] = np.abs(np.mean((za_by_factor[f] - mean_by_factor[f]) * (zb_by_factor[f] - mean_by_factor[f]), 0))
            # score_by_factor[f] = score_by_factor[f] / var
            # score_by_factor[f] = np.where(score_by_factor[f] > 0.5, 1.0, 0.0 )
            # score_by_factor[f] = np.sum(score_by_factor[f])
        else:
            # individual_scores[f] = np.ones(za_by_factor[0].shape[0])
            score_by_factor[f] = 1.0
            # why is residual factor explicitly in data?
            # why not just append it and use all the data

    scores = np.array([score_by_factor[f] for f in range(args.n_factors)])

    # SOFTMAX
    m = np.max(scores)
    e = np.exp(scores-m)
    softmaxed = e / np.sum(e)
    dim = za.shape[1]
    dims = [int(s*dim) for s in softmaxed]
    dims[-1] = dim - sum(dims[:-1])
    dims_percent = dims.copy()
    for i in range(len(dims)):
        dims_percent[i] = round(100*(dims[i] / sum(dims)), 1)
    return dims, dims_percent
