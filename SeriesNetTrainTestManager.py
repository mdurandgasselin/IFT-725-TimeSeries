import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


class TrainTestManager(object):
    """
    Class used the train and test the given model in the parameters
    """

    def __init__(self, models,
                 trainset,
                 testset,
                 lr=0.001,
                 loss_fn = 'MeanSquared',
                 optimizer_type='sgd',
                 batch_size=1):

        self.model = models[0]
        self.pre_trained_model = models[1]
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.optimizer = None
        self.batch_size= batch_size
        #self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
        #self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=False)
        self.dataset_train = trainset
        self.dataset_test = testset
        self._init_loss(loss_fn)
        self._init_optim()
        self.metrics = {}
        self.metrics['Training_Loss'] = []
        self.metrics['Testing_Loss'] = []

    def _init_optim(self):
        if self.optimizer_type == 'sgd' or self.optimizer_type == 'SGD':
            self.optimizer= optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer_type  == 'Adam' or self.optimizer_type  == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _init_loss(self, loss):
        """
        :type loss: str
        """
        if loss == 'MeanSquared' or loss == 'MSE':
            self.criterion = nn.MSELoss()



    def pretrain(self, num_epochs, pts_2pred):
        """ Pre_entrainement du model sur les derniers points de toute les courbe du dataset d'entrainement.
        On essaye de capturer la tendance de toute les courbes pour les  pts_2pred de la fin  """
        self.metrics['Training_Loss'] = []
        self.metrics['Testing_Loss'] = []
        batch_size = 10
        train_loader = DataLoader(self.dataset_train, batch_size= batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True, drop_last=False)

        P = pts_2pred
        C, L = self.dataset_train[0].size()

        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_losses = []
            for i, batch in enumerate(train_loader):
                data, target = batch[:, :, :], batch[:, :, -P:]
                losses = []
                for k in range(P):
                    #pred = torch.zeros(10, 5, 1)
                    #input_data = inputModel[:, :, :L -P]
                    output = self.model(data[:,:,:-P+k])
                    cible = target[:, :, k]
                    self.optimizer.zero_grad()
                    # On calcul la loss par rapport a la derniere prediction
                    loss = self.criterion(output[:, :, -1], cible)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

                train_losses.append(np.array(losses))
            abc=np.array(train_losses)
            self.metrics['Training_Loss'].append(np.mean(abc,axis=0))

            self.model.eval()
            with torch.no_grad():
                loss_test = []
                for i, batch in enumerate(test_loader):
                    data, target = batch, batch[:, :, -P:]
                    #input_data = X[:, :, i:start + i]
                    pred=torch.zeros(batch_size, 5, 1)
                    losses = []
                    for p in range(P):
                        input_data = self.model(data[:,:,:-P +p])
                        loss = self.criterion(input_data[:, :, -1], target[:, :, p])
                        #pred = torch.cat((pred, input_data[:, :, -1].view(batch_size, 5, 1)), 2)
                        losses.append(loss.item())
                    # On calcul la loss par rapport a la derniere prediction
                    #loss = self.criterion(pred[:, :, 1:], target)
                    loss_test.append(np.array(losses))
            abc=np.array(loss_test)
            self.metrics['Testing_Loss'].append(np.mean(loss_test,axis=0))
            self.model.train()
        #save model to pretrained file
        self.pre_trained_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.plot_preTrain_metrics()

    def train(self, epochs, data_index, pts_2pred = 10):
        """ Strategie ou on part d'un modèle entrainé et on entraine ce modèle sur les données d'une actions"""
        self.metrics['Training_Loss'] = []
        self.metrics['Testing_Loss'] = []

        #stockNb = random.randint(0, 399)
        self.model.load_state_dict(copy.deepcopy(self.pre_trained_model.state_dict()))
        self._init_optim()
        P = pts_2pred
        print(type(P))
        ind = data_index

        C, L = self.dataset_train[ind].size()
        print(type(C), type(L))
        start = self.model.get_pts_for_Pred()
        print(start)
        #print(L - P - start)
        epoch = 0
        data = self.dataset_train[ind].view(1,C,-1)
        for _ in tqdm(range(epochs)):
            print("Epoch: {} of {}".format(epoch + 1, epochs))
            epoch+=1
            train_losses = []
            for k in range(L - P - start):
                #pred = torch.zeros(10, 5, 1)
                input_data = data[:, :, k:start + k]
                output = self.model(input_data)
                cible = data[:, :, start + k ]
                self.optimizer.zero_grad()
                # On calcul la loss par rapport a la derniere prediction
                loss = self.criterion(output[:, :, -1], cible)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            self.metrics['Training_Loss'].append(np.array(train_losses))

            self.model.eval()
            prediction = torch.zeros(1, C, 1)
            with torch.no_grad():
                input_data = data[:,:,:-P]
                targets = data[:,:,-P:]
                loss_test = []
                for i in range(P):
                    output = self.model(input_data)
                    input_data = torch.cat((input_data,output[:,:,-1].view(1, C, 1)),2)
                    #prediction = torch.cat((prediction, input_data[:, :, -1].view(1, C, 1)), 2)
                    loss_eval = self.criterion(output[:,:,-1].view(1,C,1), targets[:,:,i].view(1,C,1))
                    loss_test.append(loss_eval.item())
                self.metrics['Testing_Loss'].append(np.array(loss_test))
            self.model.train()
        self.plot_Train_metrics()

    def train_1company(self, epochs, data_index, pts_2pred = 10):
        """ Strategie ou on part d'un modèle entrainé et on entraine ce modèle sur les données d'une actions"""
        self.metrics['Training_Loss'] = []
        self.metrics['Testing_Loss'] = []

        #stockNb = random.randint(0, 399)
        #self.model.load_state_dict(copy.deepcopy(self.pre_trained_model.state_dict()))
        #self._init_optim(self.optimizer,self.lr)
        P = pts_2pred
        print(type(P))
        ind = data_index

        C, L = self.dataset_train[ind].size()
        print(type(C), type(L))
        start = self.model.get_pts_for_Pred()
        print(start)
        epoch = 0
        data = self.dataset_train[ind].view(1,C,-1)
        for _ in tqdm(range(epochs)):
            print("Epoch: {} of {}".format(epoch + 1, epochs))
            epoch+=1
            train_losses = []
            for k in range(L - P - start):
                #pred = torch.zeros(10, 5, 1)
                input_data = data[:, :, k:start + k]
                output = self.model(input_data)
                cible = data[:, :, start + k ].view(1,C,1)
                self.optimizer.zero_grad()
                # On calcul la loss par rapport a la derniere prediction
                loss = self.criterion(output[:, :, -1].view(1,C,1), cible)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            self.metrics['Training_Loss'].append(np.array(train_losses))

            self.model.eval()
            prediction = torch.zeros(1, C, 1)
            with torch.no_grad():
                input_data = data[:,:,:-P]
                targets = data[:,:,-P:]
                loss_test = []
                for i in range(P):
                    output = self.model(input_data)
                    input_data = torch.cat((input_data,output[:,:,-1].view(1, C, 1)),2)
                    #prediction = torch.cat((prediction, input_data[:, :, -1].view(1, C, 1)), 2)
                    loss_eval = self.criterion(output[:,:,-1].view(1,C,1), targets[:,:,i].view(1,C,1))
                    loss_test.append(loss_eval.item())
                self.metrics['Testing_Loss'].append(np.array(loss_test))
            self.model.train()
        self.plot_Train_metrics()

    def plot_metrics(self):
        print(len(self.metrics['Training_Loss']))
        epochs = range(1, len(self.metrics['Training_Loss']) + 1)

        plt.figure(figsize=(10, 5))
        # loss plot
        plt.plot(epochs, self.metrics['Training_Loss'], '-o', label='Training loss')
        plt.plot(epochs, self.metrics['Testing_Loss'], '-o', label='Validation loss')
        plt.title('Training and testing loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        #f.savefig(figname + '.png')
        plt.show()

    def plot_Train_metrics(self):
        train_loss = np.array(self.metrics['Training_Loss'])
        test_loss = np.array(self.metrics['Testing_Loss'])
        epochs = train_loss.shape[0]
        xAxis_train = np.arange(train_loss.shape[1])
        xAxis_test = np.arange(test_loss.shape[1])
        plt.figure()
        plt.subplot(2, 1, 1)
        for epoch in range(epochs):
            plt.plot(xAxis_train, train_loss[epoch], label='epoch {}'.format(epoch + 1))
        plt.xlabel('Number ')
        plt.ylabel('Training loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        for epoch in range(epochs):
            plt.plot(xAxis_test,  test_loss[epoch], label='epoch {}'.format(epoch + 1))
        plt.xlabel('Epoch')
        plt.ylabel('Testing loss')
        plt.legend()

    def plot_preTrain_metrics(self):
        train_loss = np.array(self.metrics['Training_Loss'])
        test_loss = np.array(self.metrics['Testing_Loss'])
        epochs = train_loss.shape[0]
        xAxis = np.arange(train_loss.shape[1])
        plt.figure()
        plt.subplot(2,1,1)
        for epoch in range(epochs):
            plt.plot(xAxis, train_loss[epoch], label='epoch {}'.format(epoch+1))
        plt.xlabel('Epoch')
        plt.ylabel('Training loss')
        plt.legend()
        plt.subplot(2,1,2)
        for epoch in range(epochs):
            plt.plot(xAxis, test_loss[epoch], label='epoch {}'.format(epoch + 1))
        plt.xlabel('Epoch')
        plt.ylabel('Testing loss')
        plt.legend()

    def plot_prediction(self, data_index, axes, pts_2pred = 10):
        P = pts_2pred
        ind = data_index
        C, L = self.dataset_train[ind].size()
        data = self.dataset_train[ind].view(1,C,-1)
        cible = data[:, :, L-P:].view(C,-1).detach().numpy()
        print(cible.shape)
        self.model.eval()

        prediction = torch.zeros(1, C, 1)
        with torch.no_grad():
            input_data = data[:,:,:L-P]
            for i in range(P):
                input_data = self.model(input_data)
                prediction = torch.cat((prediction, input_data[:, :, -1].view(1, C, 1)), 2)
            #loss_eval = self.criterion(prediction[:, :, 1:], data[:, :, L - P:])
            #self.metrics['Testing_Loss'].append(loss_eval.item())
        self.model.train()
        prediction = prediction[:, :, 1:].detach().numpy()[0]
        abscisse = np.arange(0, 1259)
        data = data[:,:,:L-P].detach().numpy()[0]
        # print(abscisse.shape, data[1].shape)
        plt.subplot(211)
        plt.plot(abscisse[:-P], data[axes], label='data')
        plt.plot(abscisse[-P:], prediction[axes], label='prediction')
        plt.plot(abscisse[-P:], cible[axes], label='Cible')
        plt.xlim(1100, 1260)
        #plt.ylim(-3, 3)
        plt.legend()
        plt.subplot(212)
        plt.plot(abscisse[:-P], data[axes], label='data')
        plt.plot(abscisse[-P:], prediction[axes], label='prediction')
        plt.plot(abscisse[-P:], cible[axes], label='Cible')

        # plt.xlim(1200,1260)
        plt.legend()
        plt.show()

