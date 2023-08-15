import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class EBDDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.wordlens = [len(x) for x in dataset]

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        data_len = len(self.dataset)
        return data_len

    @staticmethod
    def padded(x, seq_len, pad_ebd=[]):
        len_x = len(x)
        for i in range(seq_len - len_x):
            x.append(pad_ebd)
        return x

class cosSim(nn.Module):
    def __init__(self):
        super(cosSim, self).__init__()

    def forward(self, a, b, eps=1e-8):
        #     a = torch.randn(2, 2)
        #     b = torch.randn(3, 2) # different row number, for the fun

        # Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
        #                          = dot(u / norm(u), v / norm(v))
        # We fist normalize the rows, before computing their dot products via transposition:
        #         added eps for numerical stability
        a_n = a.norm(dim=1)[:, None]
        b_n = b.norm(dim=1)[:, None]
        a_eps = eps * torch.ones_like(a_n)
        b_eps = eps * torch.ones_like(b_n)
        if a_n.is_cuda:
            a_eps = a_eps.cuda()
            b_eps = b_eps.cuda()
        a_norm = a / torch.max(a_n, a_eps)
        b_norm = b / torch.max(b_n, b_eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


class CosSimWrapper():
    def __init__(self):
        self.use_cuda = False
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            self.model = cosSim().cuda()
            self.use_cuda = True
        else:
            self.model = cosSim().cpu()

    def max_cos_sim_each_token(self, a, b, batch_szie=2048):
        # a is a list of n_samples. each sample is a list of n_words embedding with size n_features.
        # b is a list of n_words embedding with size n_features.
        # return a list[list], in which each value is a similarity
        res = []

        a = EBDDataset(a)
        b = EBDDataset(b)

        load_a = DataLoader(a, batch_size=batch_szie, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=0, collate_fn=lambda x: x)

        load_b = DataLoader(b, batch_size=batch_szie, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=0, collate_fn=lambda x: x)

        sim_list = []
        for a_batch in load_a:
            # 先让样本等长
            n_samples = len(a_batch)
            seq_lens = [len(alist) for alist in a_batch]
            ebd_len = len(a_batch[0][0])
            ept_ebd = [0] * ebd_len
            a_batch = [EBDDataset.padded(alist, max(seq_lens), pad_ebd=ept_ebd) for alist in a_batch]

            # 变成n_samples*n_words, ebd_len
            a_batch = np.array(a_batch).reshape(-1, ebd_len)

            i_sim_list = []
            for b_batch in load_b:
                a_batch = torch.tensor(a_batch).clone().detach().requires_grad_(False).float()
                b_batch = torch.tensor(b_batch).clone().detach().requires_grad_(False).float()

                if self.use_cuda:
                    a_batch = a_batch.cuda()
                    b_batch = b_batch.cuda()

                sim_mat = self.model(a_batch, b_batch)
                torch.cuda.empty_cache()

                i_sim_list.append(sim_mat.cpu().tolist())

            i_sim_list = np.concatenate(i_sim_list, axis=1).reshape(n_samples, max(seq_lens), -1).tolist()

            i_sim_list = [alist[:seq_lens[ith]] for ith, alist in enumerate(i_sim_list)]
            # max similarity
            i_sim_list = [[max(alist_inside) for alist_inside in alist] for alist in i_sim_list]
            sim_list.extend(i_sim_list)
        return sim_list

    @staticmethod
    def get_batches(a, batch_size=2):
        a_batches = []
        for i in range(int(len(a) / batch_size) + 1):
            if i * batch_size < len(a):
                a_batches.append(a[i * batch_size:i * batch_size + batch_size])
        return a_batches

    def cos_sim(self, a, b, batch_size=1024):
        a_batches = CosSimWrapper.get_batches(a, batch_size=batch_size)
        b_batches = CosSimWrapper.get_batches(b, batch_size=batch_size)

        sim_list = []
        for ith, a_batch in enumerate(a_batches):
            i_sim_list = []
            a_batch = torch.tensor(a_batch).float()
            if self.use_cuda:
                a_batch = a_batch.cuda()
            for jth, b_batch in enumerate(b_batches):
                b_batch = torch.tensor(b_batch).float()
                if self.use_cuda:
                    b_batch = b_batch.cuda()
                sim_mat = self.model(a_batch, b_batch)
                i_sim_list.append(sim_mat.cpu().tolist())
                # print(ith, jth, " batch")
            i_sim_list = np.concatenate(i_sim_list, axis=1).tolist()
            sim_list.extend(i_sim_list)
        return sim_list

    def cos_sim_list(self, a, b, threshold, batch_size=1024):
        a_batches = CosSimWrapper.get_batches(a, batch_size=batch_size)
        b_batches = CosSimWrapper.get_batches(b, batch_size=batch_size)

        sim_list = [set() for ith in range(len(a))]
        start_ridx = 0
        for ith, a_batch in enumerate(a_batches):
            row_idxs = [x + start_ridx for x in range(len(a_batch))]
            i_sim_list = []
            a_batch = torch.tensor(a_batch).float()
            if self.use_cuda:
                a_batch = a_batch.cuda()

            start_cidx = 0
            for jth, b_batch in enumerate(b_batches):
                col_idxs = [x + start_cidx for x in range(len(b_batch))]
                b_batch = torch.tensor(b_batch).float()
                if self.use_cuda:
                    b_batch = b_batch.cuda()
                sim_mat = self.model(a_batch, b_batch)
                sim_mat = sim_mat.cpu().tolist()

                for iith, sims in enumerate(sim_mat):
                    for jjth, asim in enumerate(sims):
                        if asim >= threshold:
                            sim_list[row_idxs[iith]].add(col_idxs[jjth])
                del sim_mat
                del b_batch
                torch.cuda.empty_cache()
                start_cidx += len(col_idxs)
            start_ridx += len(row_idxs)
        return sim_list

    @staticmethod
    def max_cos_sim(raw_a, raw_b, raw_tokens_train, sim_mat):
        # raw_tokens_dict: {token:idx}
        # sim_mat = self.cos_sim(tokens_ebd, tokens_ebd, batch_size=batch_size)
        token_idx_dict = {token: i for i, token in enumerate(raw_tokens_train)}
        sim_list = []
        for ith, a_seq in enumerate(raw_a):
            i_sim_list = []
            for jth, a_token in enumerate(a_seq):
                idx_a = token_idx_dict[a_token]
                sim_with_b = []
                for b_token in raw_b:
                    idx_b = token_idx_dict[b_token]
                    sim = sim_mat[idx_a][idx_b]
                    sim_with_b.append(sim)
                if len(sim_with_b) == 0:
                    i_sim_list.append(0)
                else:
                    i_sim_list.append(max(sim_with_b))
            sim_list.append(i_sim_list)
        return sim_list
