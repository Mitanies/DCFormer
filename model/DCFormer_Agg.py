import torch
import torch.nn as nn
# from model.module.trans import Transformer as Transformer_s
# from model.module.trans_hypothesis import Transformer
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp

from timm.models.layers import DropPath
from model.block.vanilla_transformer_encoder import Transformer 


class Mlp(nn.Module):
    def __init__(self, args=None, in_features=3, hidden_features=256, out_features=3, refine=False):
        super().__init__()

        if args == None:
            out_channels, in_channels, n_joints = 3, 2, 17
        else:
            out_channels, in_channels, n_joints = args.out_channels, args.in_channels, args.n_joints
        self.trans_agg = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            Transformer(2, hidden_features, hidden_features, length=n_joints), 
            nn.Linear(hidden_features, out_features)
        )

        if refine:
            self.refine_agg = nn.Sequential(
                nn.Linear(out_channels*2*args.n_joints, 1024),
                nn.ReLU(),
                nn.Dropout(0.5,inplace=False),
                nn.Linear(1024, in_channels*args.n_joints),
                nn.Sigmoid()
            )

    def forward(self, xy, z, split=None, uvd=None, refine=False):

        b, f, n, _  = xy.shape
        x_in = torch.cat((xy, z),dim=-1).reshape(-1, n, 3)

        x_out = self.trans_agg(x_in).reshape(b, f, n, -1)
        x_out = torch.cat((xy,x_out[:,:,:,-1].unsqueeze(-1)),dim=-1)

        if refine:
            refine_in = torch.cat((x_out, uvd), -1).view(b, -1)
            score = self.refine_agg(refine_in).view(b, f, n, 2)
            score_cm = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score
            x_out[:,:,:,:2] = score * x_out[:,:,:,:2] + score_cm * uvd[:,:,:,:2]

        return x_out

if __name__ == "__main__":
    # inputs = torch.rand(64, 351, 34)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = Mlp()
    inputs_xy = torch.rand([1, 27, 17, 2])
    inputs_z = torch.rand([1, 27, 17, 1])
    #output = net(inputs)
    #print(output.size())
    from thop import profile

    flops, params = profile(net, inputs=(inputs_xy,inputs_z))
    print(flops)
    print(params)
