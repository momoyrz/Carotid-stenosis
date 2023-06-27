import scipy.sparse as sp
import copy
import torch

from torch import nn
import torch.nn.functional as F

from .layers import *

class MRGCN(nn.Module):
    """
    multi-relational GCN
    """
    def __init__(self, relations, in_dim=None, out_dim=None, enc='alex', inchannel=3, share_encoder='partly', selfweight=1):
        super(MRGCN, self).__init__()
        self.selfweight = selfweight
        if enc == 'enet':
            self.encoder = MyEfficientB0()
        elif enc == 'singlealex':
            self.encoder = MyAlexNet(inchannel=inchannel).features
        elif enc == 'resnet34':
            self.encoder = MyResNet34()
        elif enc=='singleres50':
            self.encoder = MyResNet50(inchannel=inchannel).features
        elif enc=='singleres50_1':
            self.encoder = MyResNet50(inchannel=inchannel).features
        elif enc=='singledens161':
            self.encoder=MyDensNet161(inchannel=inchannel).features
        elif enc=='singledens201':
            self.encoder=MyDensNet201(inchannel=inchannel).features
        elif enc=='singledens121':
            self.encoder=MyDensNet121(inchannel=inchannel).features

        self.share_encoder = share_encoder
        if share_encoder == 'totally':
            self.gcn = ImageGraphConvolution(self.encoder, out_dim=out_dim, inchannel=inchannel) if enc else GraphConvolution(in_dim, out_dim)
        elif share_encoder == 'not':
            self.gcn = nn.ModuleDict({str(i): ImageGraphConvolution(enc=copy.deepcopy(self.encoder), out_dim=out_dim, inchannel=inchannel)  \
                                      for i in relations+['self']}) if enc else  \
                                    nn.ModuleDict({str(i): GraphConvolution(in_dim, out_dim) for i in relations+['self']})
        elif share_encoder == 'partly':
            self.gcn = nn.ModuleDict({str(i): GraphConvolution(in_dim, out_dim, singal=i) for i in relations+['self']})

    def forward(self, fea_in, k, adj_mats ):
        if self.share_encoder == 'totally':
            adj = sum(adj_mats.values())
            adj[:,k:] += self.selfweight * torch.eye(len(fea_in)-k).cuda()
            fea = self.gcn(fea_in , adj)
        elif self.share_encoder == 'not':
            fea_out = fea_in[k:]
            fea = self.gcn['self'](fea_out, self.selfweight)
            for i, adj in adj_mats.items():
                fea = fea + self.gcn[i](fea_in, adj)
        elif self.share_encoder == 'partly':
            fea_in = self.encoder(fea_in).squeeze()
            fea_out = fea_in[k:]
            fea = self.gcn['self'](fea_out, self.selfweight)
            for i, adj in adj_mats.items():
                fea = fea + self.gcn[i](fea_in, adj)

        return fea
    
class SingleLayerImageGCN(nn.Module):
    def __init__(self, relations, encoder='resnet34', in_dim=1280, out_dim=4, inchannel=3, share_encoder='partly'):
        super(SingleLayerImageGCN, self).__init__()
        self.out_dim = out_dim
        self.layer = MRGCN(relations, enc=encoder, in_dim=in_dim, out_dim=out_dim, inchannel=inchannel, share_encoder=share_encoder )
    
    def forward(self, fea,  adj_mats, k):
        fea2 = self.layer(fea, k, adj_mats)
        fea2 = fea2.view(-1, self.out_dim)

        return fea2
    


