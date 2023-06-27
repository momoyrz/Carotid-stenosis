import pandas as pd
import numpy as np
import pickle as pk
import copy
import gc
from torch.utils.data import Dataset, DataLoader
from os.path import join
from PIL import Image
from .preprocessing import adj_from_series
from .selfdefine import FlexCounter
from heapq import nlargest
from collections import Counter


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pk.load(f)


class Chexpert_Dataset(Dataset):
    """Chexpert dataset."""

    def __init__(self, args, path='/home/ubuntu/qujunlong/data/yuanshi',
                 mode='RGB',
                 adjgroup=True,
                 neib_samp='relation',
                 relations=None,
                 k=3,
                 graph_nodes='current',
                 transform_fn=None,
                 ):
        self.classes = ['0', '1', '2', '3']
        self.label = None
        self.args = args
        self.path = path
        self.all_label_df = pd.read_csv(
            '/home/ubuntu/qujunlong/huhuiling_4class/GCN/feature_del121_fillna_after_standard_scaler.CSV',
            encoding='gbk')
        self.mode = mode
        self.adjgroup = adjgroup
        self.neib_samp = neib_samp
        self.k = k
        self.gnode = graph_nodes
        self.transform = transform_fn
        self.relations = relations
        self.label_df = self.all_label_df
        self.all_grp = self.creat_adj(self.label_df)
        self.grp = self.all_grp

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        sample = self._getimage(idx)
        if self.neib_samp == 'relation':
            img = self.label_df.iloc[idx]
            impt = self.impt_sample(img, k=self.k)
            sample['impt'] = impt
        return sample

    def _getimage(self, idx, byindex=False, level=0):
        img = self.all_label_df.loc[idx] if byindex else self.label_df.iloc[idx]
        image = Image.open(join(self.path, img['image'])).convert(self.mode)
        labels = img[1] - 5

        sample = {'image': image, 'label': labels, 'pid': img[2], 'age': img['age'], 'sex': img['sex'],
                  'Separation of eyes': img['Separation of eyes'], 'name': img['image'], 'index': img.name}
        if level == 0:
            sample['dataset'] = self
            if self.neib_samp in ('sampling', 'best'):
                w = sum([(FlexCounter(grp[img[r]]) / len(grp[img[r]]) if img[r] in grp else FlexCounter())
                         for r, grp in self.tr_grp.items()], Counter())
                sample['weight'] = w

        # if self.label == 'train':
        #     sample['image'] = self.transform['train'](sample['image'])
        # else:
        #     sample['image'] = self.transform['test'](sample['image'])
        sample['image'] = self.transform(sample['image'])
        return sample

    def impt_sample(self, img, method='relation', k=1, base='train'):
        """
        sampling the important k samples for img.
        method: "sample"--random choose by probability
                "best"--choose the most important
                "relation"--random choose k for each relation
        base: choose the basic set, "train" or "all"
        """
        if base == "train":
            grps = self.tr_grp
        elif base == "all":
            grps = self.all_grp

        if method == 'relation':
            impt_sample = []
            for r, grp in grps.items():
                if img[r] in grp:
                    neibs = grp[img[r]].drop(img.name, errors='ignore')
                    if not neibs.empty:
                        # k = min(k, len(neibs))
                        impt_sample += np.random.choice(neibs, min(k, len(neibs)), replace=False).tolist()
            return impt_sample

        w = sum([FlexCounter(grp[img[r]]) / len(grp[img[r]]) for r, grp in grps.items()], Counter())
        w.pop(img.name, None)
        if method == "sample":
            p = FlexCounter(w) / sum(w.values())
            impt_sample = np.random.choice(list(p.keys()), k, replace=False, p=list(p.values()))
        elif method == 'best':
            impt_sample = nlargest(k, w, key=w.get)

        return impt_sample

    def creat_adj(self, label_df, adjgroup=True):
        if self.gnode == 'current':
            adj = {r: adj_from_series(label_df[r], groups=adjgroup) for r in self.relations}
        else:
            pass
        return adj

    def tr_val_te_split(self, split='random', tr_pct=0.7):
        n_all = len(self.all_label_df)
        np.random.seed(0)
        if split == 'specified':
            perm = np.random.permutation(223414)
            tr_df = self.all_label_df.iloc[perm[:int(223414 * 0.9)]]
            val_df = self.all_label_df.iloc[perm[int(223414 * 0.9):]]
            te_df = self.all_label_df.iloc[-234:]
        elif split == 'random':
            tr_df, val_df, te_df = np.split(self.all_label_df.sample(frac=1, random_state=0),
                                            [int(n_all * 0.7), int(n_all * 0.9)])
            tr_df = tr_df.sample(n=int(n_all * tr_pct), random_state=0)

        self.tr_grp = self.creat_adj(tr_df)

        tr_set, val_set, te_set = copy.copy(self), copy.copy(self), copy.copy(self)
        tr_set.label_df, val_set.label_df, te_set.label_df = tr_df, val_df, te_df
        tr_set.label, val_set.label, te_set.label = 'train', 'val', 'test'

        return tr_set, val_set, te_set
