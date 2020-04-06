from typing import Any

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import glob
from torch.utils.data import Dataset

class ActionDataset(Dataset):
    """FN-500 action dataset."""

    def __init__(self, root_dir, normalise = True):
        """
        Args:
            root_dir (string): Directory with all the actions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #print(root_dir + "/*.csv")
        all_files = glob.glob(root_dir + "/*.csv")
        #print(all_files)
        li = []
        for i, filename in enumerate(all_files):
            act = pd.read_csv(filename)
            if act.shape[0] == 1259:
                col = act.columns[1:-1]
                if (np.isnan(act[col].values)).sum() != 0 :
                    print(i)
                    print('problemo : ', filename)
                    print(act[col].isnull())
                    A = act[col]
                    print(A[pd.isnull(A).any(axis=1)])  #  df[pd.isnull(df).any(axis=1)]
                x = np.transpose(act[col].values, (1, 0))
                li.append(x)
        print("lenght : ", len(li))
        li = np.stack(li, axis=0)
        if normalise:
            aa = torch.tensor(li)
            print(aa.size())
            self.actions = (aa - aa.mean(axis=2).reshape(470, 5, 1)) / aa.std(axis=2).reshape(470, 5, 1)
        else:
            self.actions = torch.tensor(li)  # (470, 5, 1259)
            print("action.size ", self.actions.size())

    def __len__(self):
        return self.actions.size()[0]

    def __getitem__(self, idx):
        return self.actions[idx]

    def float(self):
        self.actions = self.actions.float()

    def double(self):
        self.actions = self.actions.double()
