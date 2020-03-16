# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 08:19:14 2020

@author: max-d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class seriesNet(nn.Module):

    def __init__(self, in_channels, out_channels, gate_nb_filter=32 ):
        super(seriesNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel parametrisation
        #self.batch_norm_data = nn.BatchNorm1d(in_channels)
        self.gated_block1 = gated_block(in_channels, gate_nb_filter, dilation=1 )
        self.gated_block2 = gated_block(in_channels, gate_nb_filter, dilation=2 )
        self.gated_block3 = gated_block(in_channels, gate_nb_filter, dilation=4 )
        self.gated_block4 = gated_block(in_channels, gate_nb_filter, dilation=8 )
        self.gated_block5 = gated_block(in_channels, gate_nb_filter, dilation=16 )
        
        self.gated_block6 = gated_block(in_channels, gate_nb_filter, dilation=32)
        self.drop_1 = nn.Dropout(p=0.35)
        self.gated_block7 = gated_block(in_channels, gate_nb_filter, dilation=64 )
        self.drop_2 = nn.Dropout(p=0.35)
        
        self.conv_final = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size= 1, stride=1, padding=0, bias=False)
        

    def forward(self, x):
        
        #x = self.batch_norm_data(x)
        x, skip1 = self.gated_block1(x)
        x, skip2 = self.gated_block2(x) 
        x, skip3 = self.gated_block3(x)
        x, skip4 = self.gated_block4(x)
        x, skip5 = self.gated_block5(x)
        x, skip6 = self.gated_block6(x)
        
        skip6 = self.drop_1(skip6) #dropout used to limit influence of earlier data
        
        x, skip7 = self.gated_block6(x)
        skip7 = self.drop_2(skip7) #dropout used to limit influence of earlier data

        output =   skip1 + skip2 + skip3 + skip4 + skip5 + skip6 + skip7
        
        output =   F.relu(output)
        
        output = self.conv_final(output)
        
        return output



class gated_block(nn.Module):
    # Param : nb_filter, filter_length, dilation, l2_layer_reg
    def __init__(self, in_channels, nb_filter, dilation=1 ):
        super(gated_block, self).__init__()
        
        self.pad_input = nn.ReflectionPad1d((dilation, 0))
        
        #self.batch_N_lay_out = nn.BatchNorm1d(nb_filter)

        #self.batch_N_skipout = nn.BatchNorm1d(in_channels)
                  
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=nb_filter,
                              kernel_size= 2, stride=1, padding=0,
                              dilation=dilation, bias=False, padding_mode='reflect')
        
        self.network_in = nn.Conv1d(in_channels=nb_filter, out_channels=in_channels, kernel_size= 1,
                                    stride=1, padding=0, dilation=1, groups=1,
                                    bias=False)
        
        self.skipout = nn.Conv1d(in_channels=nb_filter, out_channels=in_channels,
                                 kernel_size= 1, stride=1, padding=0, dilation=1,
                                 bias=False)
    
    def forward(self, x):
                
        residual = x
        
        #x = F.selu(self.batch_N_lay_out(self.conv(self.pad_input(x))))
        x = F.selu(self.conv(self.pad_input(x)))
        
        skip = self.skipout(x)
        net_in = self.network_in(x)
        
        layer_out = residual + net_in
    
        return layer_out, skip
    

#m = nn.Conv1d(6, 33, 3, stride=1,padding=0, padding_mode='replicate')
#input = torch.randn(20, 6, 1223)
#output = m(input)
#print(output.size())
