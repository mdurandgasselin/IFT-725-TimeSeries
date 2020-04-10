import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


class TrainManager(object):
    """
    Class used the train and test the given model in the parameters
    """

    def __init__(self, model,
                 train_dataLoader,
                 eval_dataloader,
                 lr=0.001,
                 loss_fn = 'MeanSquared',
                 optimizer_type='sgd',
                 pts_2pred = 10):

        self.model = model
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.train_dataLoader = train_dataLoader
        self.eval_dataLoader = eval_dataloader
        self.criterion = self.init_loss(loss_fn)
        self.optimizer = self.init_optim()
        self.metrics_epoch = {'Training_Loss': [], 'Validation_Loss': []}

    def init_optim(self):
        if self.optimizer_type == 'sgd' or self.optimizer_type == 'SGD':
                opt = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer_type == 'Adam' or self.optimizer_type == 'adam':
                opt = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            opt = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        return opt

    def init_loss(self, loss):
        """
        :type loss: str
        """
        if loss == 'MeanSquared' or loss == 'MSE':
            crit = nn.MSELoss()
        else:
            crit = nn.MSELoss()
        return crit

    def train(self, epoch, display=True, figname='fig' ):
        #loss_epoch = []
        #loss_eval_epoch = []
        for e in range(epoch):
            #print('epoch {}/ {}'.format(e,epoch))
            losses = []
            for data, target in self.train_dataLoader:
                N, C, W = data.size()
                output = self.model(data)
                self.optimizer.zero_grad()
                # loss=criterion(output, target)
                loss = self.criterion(output[:, :, -1].view(N, C, 1), target)
                # print('loss : ',loss.item())
                loss.backward()
                self.optimizer.step()
                # print(model.module_block[-2].conv.weight.grad)
                losses.append(loss.item())
            self.metrics_epoch['Training_Loss'].append(np.mean(losses))

            self.model.eval()

            with torch.no_grad():
                losses = []
                for data, target in self.eval_dataLoader:
                    N, C, W = data.size()
                    output = self.model(data)
                    loss = self.criterion(output[:, :, -1].view(N, C, 1), target)
                    losses.append(loss.item())
            self.metrics_epoch['Validation_Loss'].append(np.mean(losses))
            self.model.train()
        if display:
            self.plot_metrics(figname=figname)



    def predict(self,input, target):
        self.model.eval()
        _, C, P = target.size()
        X = input.clone()
        with torch.no_grad():
            pred = []
            for p in range(P):
                output = self.model(X)
                X = torch.cat((X[:, :, 1:], output[:, :, -1].view(1, C, 1)), axis=2)
                pred.append(output[:, :, -1].view(1).item())
        pred = np.array(pred)
        self.model.train()
        return pred

    def plot_metrics(self, figname='fig'):
        plt.figure()
        plt.plot(self.metrics_epoch['Training_Loss'], label='Training')
        plt.plot(self.metrics_epoch['Validation_Loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Moyenne')
        plt.legend()
        plt.savefig(figname+'_metric.png')
        plt.show()

    def plot_prediction(self, _input, target, pts_2_pred, save='False', figname = 'fig'):
        pred = self.predict(_input, target)
        N = self.model.get_pts_for_Pred()
        abscisse = np.arange(0, N + pts_2_pred)
        data = _input.view(N).detach().numpy()
        target = target.view(pts_2_pred).detach().numpy()

        plt.figure()
        plt.plot(abscisse[:N], data, label='Donnée')
        plt.plot(abscisse[N:], pred, label='Prédiction')
        plt.plot(abscisse[N:], target, label='Cible')
        plt.legend()
        plt.xlabel("Temps (jour)")
        plt.ylabel('Données normalisés')
        plt.savefig(figname+'_pred.png')
        plt.show()

