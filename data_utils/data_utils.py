import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from platform import system
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import warnings
warnings.filterwarnings("ignore")


class DataUtils:

    def __init__(self, window=60, company='AAPL'):

        self.company = company
        self.window = window
        self.data = None

        if system() == 'Linux':
            self.path = "dataset/sandp500/all_stocks_5yr.csv"

    def read_data(self):

        data = pd.read_csv(self.path, parse_dates=True, na_values=['nan'])
        data.loc[:, 'date'] = pd.to_datetime(data.loc[:, 'date'], format="%Y/%m/%d")
        df = data.loc[data['Name'] == self.company, :]
        df = df[['date', 'close']].sort_values(by=['date'])
        close_data = df[['close']]
        transformer = MinMaxScaler().fit(close_data.values)
        intermediate_value = transformer.transform(close_data.values)
        close_data.loc[:, 'close'] = intermediate_value
        self.data = close_data

    def get_features_target(self):
        features, target = [], []
        length = len(self.data)
        for i in range(0, (length - self.window) + 1):
            if i + self.window < length:
                features.append(self.data.values[i:i + self.window, 0])
                target.append(self.data.values[i + self.window, 0])

        features = np.array(features)
        features, target = np.reshape(features, (features.shape[0], features.shape[1], 1)), np.array(target)
        return torch.from_numpy(features), torch.from_numpy(target)

    """
    def convert_train_val_to_tensor(self, len_val_data=199, batch_size=100):
        x, y = self.get_feature_target()
        print("x.shape", x.shape, "y.shape", y.shape)
        nb_training_data = x.shape[0] - len_val_data

        x_train, y_train = x[:nb_training_data, :, :], y[:nb_training_data]
        x_val, y_val = x[nb_training_data:, :, :], y[nb_training_data:]

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_val = torch.from_numpy(x_val).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_val = torch.from_numpy(y_val).type(torch.Tensor)

        train = torch.utils.data.TensorDataset(x_train, y_train)
        val = torch.utils.data.TensorDataset(x_val, y_val)

        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(dataset=val, batch_size=batch_size, shuffle=False)

        return train_loader, validation_loader
    """

data_utils = DataUtils()
data_utils.read_data()
features, target = data_utils.get_features_target()
print(features.size())