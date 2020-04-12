from typing import Any

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import glob
from torch.utils.data import Dataset



def create_sliding_dataset(pts_window, pts_2_pred = 10,
                           proportion = 20,
                           root_dir = "./dataset/sandp500/individual_stocks_5yr/individual_stocks_5yr/",
                           action_name='AAPL_data',
                           axis='all', normalise = True):

    act = pd.read_csv(root_dir + action_name + '.csv')

    pts_window = pts_window  # nombre de pts necessaire pour les fenetre
    pts_2_pred = pts_2_pred  # nombre de pts a predire apres entrainment

    if axis == 'all':
        col = act.columns[1:-1]
    elif axis == 'open':
        col = act.columns[1]
    elif axis == 'high':
        col = act.columns[2]
    elif axis == 'low':
        col = act.columns[3]
    elif axis == 'close':
        col = act.columns[4]
    elif axis == 'volume':
        col = act.columns[5]
    if axis != 'all':
        x = np.transpose(act[col].values.reshape(1259, 1), (1, 0))
    else:
        x = np.transpose(act[col].values, (1, 0))
    C, L = x.shape
    if normalise and axis == 'all':
        aa = torch.tensor(x).view(1, C, -1)
        action = (aa - aa.mean(axis=2).reshape(1, C, 1)) / aa.std(axis=2).reshape(1, C, 1)
    elif normalise and axis != 'all':
        aa = torch.tensor(x).view(1, C, -1)
        action = (aa - aa.mean(axis=2).reshape(1, C, 1)) / aa.std(axis=2).reshape(1, C, 1)
    else:
        action = torch.tensor(x).view(1, C, -1)  # (1, C, 1259)

    train_window, target_train, test_input, test_target = create_Window(action, pts_window, pts_2_pred)

    train_window, target_train, eval_train, eval_target = split_train_eval_sorted(train_window,
                                                                                  target_train,
                                                                                  proportion=proportion )
    dataset_train = WindowDataset(train_window, target_train)
    dataset_eval = WindowDataset(eval_train, eval_target)

    return  dataset_train, dataset_eval, test_input.float(), test_target.float()


def create_Window(action, pts_window, pts_2_pred):
    _,C,L = action.size()
    W = pts_window
    P = pts_2_pred
    data = []
    target = []
    for k in range(L - P - W):
        data.append(action[:,:,k:W+k])
        target.append(action[:,:,W+k].view(1,C,-1))
    train_window=torch.cat(data, axis=0)
    target_window=torch.cat(target, axis=0)
    test_input = action[:,:,-(P+W):-P]
    test_target = action[:,:,-P:]
    return train_window, target_window, test_input, test_target

def split_train_eval_sorted(tensor, tensor_target, proportion=20, seed = 20 ):
    N, C, W = tensor.size()
    prop = int(proportion * N / 100)
    np.random.seed(seed)
    indice = np.sort(np.random.choice(np.arange(N), prop, replace=False))
    eval_window = []
    eval_target = []
    training_window = []
    trainig_target = []
    for i in range(N):
        if i in indice:
            eval_window.append(tensor[i])
            eval_target.append(tensor_target[i])
        else:
            training_window.append(tensor[i])
            trainig_target.append(tensor_target[i])
    eval_window = torch.cat(eval_window,axis=0)
    eval_target = torch.cat(eval_target, axis=0)
    training_window = torch.cat(training_window,axis=0)
    trainig_target = torch.cat(trainig_target,axis=0)

    return training_window.view(N-prop,1,W), trainig_target.view(N-prop,C,1), eval_window.view(prop,C,W), eval_target.view(prop,C,1)

class WindowDataset(Dataset):
    """FN-500 action dataset."""

    def __init__(self,input, target ):
        """
        Args:
            root_dir (string): Directory with all the actions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input = input.float()
        self.target = target.float()

    def __len__(self):
        return self.input.size()[0]

    def __getitem__(self, idx):
        return self.input[idx],self.target[idx]

    def float(self):
        self.input = self.input.float()
        self.target = self.target.float()

class ActionDataset(Dataset):
    """FN-500 action dataset."""

    def __init__(self, root_dir, axes='all', normalise = True):
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
                if axes=='all':
                    col = act.columns[1:-1]
                elif axes == 'open':
                    col = act.columns[1]
                elif axes == 'high':
                    col = act.columns[2]
                elif axes == 'low':
                    col = act.columns[3]
                elif axes == 'close':
                    col = act.columns[4]
                elif axes == 'volume':
                    col = act.columns[5]

                if (np.isnan(act[col].values)).sum() != 0 :
                    print(i)
                    print('problemo : ', filename)
                    print(act[col].isnull())
                    A = act[col]
                    print(A[pd.isnull(A).any(axis=1)])  #  df[pd.isnull(df).any(axis=1)]
                if axes != 'all':
                    x = np.transpose(act[col].values.reshape(1259,1), (1, 0))
                else:
                    x = np.transpose(act[col].values, (1, 0))
                li.append(x)
        #print("lenght : ", len(li))
        li = np.stack(li, axis=0)
        if normalise and axes=='all':
            aa = torch.tensor(li)
            #print(aa.size())
            self.actions = (aa - aa.mean(axis=2).reshape(470, 5, 1)) / aa.std(axis=2).reshape(470, 5, 1)
        elif normalise and axes!='all':
            aa = torch.tensor(li)
            #print(aa.size())
            self.actions = (aa - aa.mean(axis=2).reshape(470, 1, 1)) / aa.std(axis=2).reshape(470, 1, 1)
        else :
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







