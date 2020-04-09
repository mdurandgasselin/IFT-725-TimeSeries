# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:09:39 2020

@author: max-d
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import glob

class action():
    """
    Représente une action
    """
    
    def __init__(self, filename):
        """ Constructeur """ 
        dirData = ".\\individual_stocks_5yr\\"
        if dirData not in filename:
            self.data = pd.read_csv(dirData + filename)
        else:
            self.data = pd.read_csv(filename)
        
        #self.data.insert(1,
        #                 "datetime",
        #                pd.to_datetime(self.data['date'],   format='%Y-%m-%d'))
        self.data.index = pd.to_datetime(self.data.date).dt.date
        
    def plot(self, attributes):
        self.data.plot(y=attributes)
        plt.xticks(rotation='vertical')
    
    def shape(self):
        return self.data.shape
    
    def to_tensor(self):
        colone= self.data.columns[1:-1]
        return torch.transpose(torch.tensor(self.data[colone].values), 0,1)
         

def load_dataset(pathName):
    
    all_files = glob.glob(pathName + "/*.csv")
    print(len(all_files))
    li = []

    for filename in all_files:
        act = action(filename)
        if act.shape()[0] == 1259:
            li.append(act.to_tensor())
    
    li = np.stack(li, axis=0)
    
    return torch.tensor(li)
    
    
    
              