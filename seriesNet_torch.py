# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 08:19:14 2020

@author: max-d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class seriesNet(nn.Module):

    def __init__(self, in_channels,  nb_causal_block=7, gate_nb_filter=32,
                            nb_block_dropped=2, dropout_rate=0.7):
            super(seriesNet, self).__init__()

            assert nb_block_dropped < nb_causal_block

            self.pts_need_for_pred = int(np.sum((2*np.ones(nb_causal_block)) ** np.arange(nb_causal_block)) +1)
            self.nb_block = nb_causal_block
            self.nb_block_dropped = nb_block_dropped

            self.module_block = nn.ModuleList()
            for i in range(self.nb_block):
                self.module_block.append(gated_block(in_channels, gate_nb_filter, dilation=2 ** i))

            self.drop_layer = nn.ModuleList()
            for j in range(self.nb_block_dropped):
                self.drop_layer.append(nn.Dropout(p=dropout_rate))

            self.conv_final = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        output = torch.zeros_like(x)
        for i in range(self.nb_block):
            x, skip = self.module_block[i](x)
            if i >= (self.nb_block - self.nb_block_dropped):
                skip = self.drop_layer[i - (self.nb_block - self.nb_block_dropped)](skip)
            output += skip

        output = F.relu(output)

        output = self.conv_final(output)
        
        return output

    def get_pts_for_Pred(self):
        return self.pts_need_for_pred



class gated_block(nn.Module):
    # Param : nb_filter, filter_length, dilation, l2_layer_reg
    def __init__(self, in_channels, nb_filter, dilation=1):
        super(gated_block, self).__init__()
        self.pad_input = nn.ReflectionPad1d((dilation, 0))
        
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=nb_filter,
                              kernel_size= 2, stride=1, padding=0,
                              dilation=dilation, bias=True, padding_mode='reflect')
        
        self.network_in = nn.Conv1d(in_channels=nb_filter, out_channels=in_channels, kernel_size= 1,
                                    stride=1, padding=0, dilation=1, groups=1,
                                    bias=True)
        
        self.skipout = nn.Conv1d(in_channels=nb_filter, out_channels=in_channels,
                                 kernel_size= 1, stride=1, padding=0, dilation=1,
                                 bias=True)
    
    def forward(self, x):
                
        residual = x
        x = F.selu(self.conv(self.pad_input(x)))
        skip = self.skipout(x)
        net_in = self.network_in(x)
        
        layer_out = residual + net_in
    
        return layer_out, skip
    

