import os, sys
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from data_utils.data_utils import DataUtils


class LSTMDataSet(Dataset):

    def __init__(self, window=60, company='AAPL'):
        self.data_utils = DataUtils(window=window)
        self.data_utils.read_data()
        self.X, self.Y = self.data_utils.get_features_target()

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.Y.size(0)
