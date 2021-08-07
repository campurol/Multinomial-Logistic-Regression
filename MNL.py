import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import math
# serialization and deserialization of model
import pickle

import numpy as np
import pandas as pd

class MNL(nn.Module):
    '''
        The Multinomial Logistic Regression model implemented with Pytorch
    '''
    def __init__(self, feature_list):
        super().__init__()   #MNL, self

        self.feature_list = feature_list
        input_dim = len(feature_list)
        # a linear layer without bias
        self.linear = torch.nn.Linear(input_dim, 1, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x, ids):
        #expect the input to be many sessions of alternatives (within a batch)
        #session_id included
        #calculate the softmax within a session

        # expect the input to be a session of alternatives
        util_values = self.linear(x)

        #!! a trik to prevent the underflow (divide by zero) in the softmax later
        util_values = util_values + 2

        # transpose the result vector before the softmax
        util_values = torch.t(util_values)
        ids=torch.t(ids)
        
        # Softmax utilities by session_id
        idxs, vals = torch.unique(ids,return_counts=True)
        vs = torch.split(util_values,tuple(vals),dim=1)
        d = [self.softmax(v) for k,v in zip(idxs,vs)]
        softmaxfx = torch.cat(d, dim=1)

        return torch.t(softmaxfx)


    def l1_loss(self, l1_weight=0.01):
        '''
            Return: L1 regularization on all the parameters
        '''
        params_list = []
        for param in self.parameters():
            params_list.append(param.view(-1))
        torch_params = torch.cat(params_list)

        return l1_weight * (torch.abs(torch_params).sum())


    def l2_loss(self, l2_weight=0.01):
        '''
            Return: L2 regularization on all the parameters
        '''
        params_list = []
        for param in self.parameters():
            params_list.append(param.view(-1))
        torch_params = torch.cat(params_list)

        return l2_weight * (torch.sqrt(torch.pow(torch_params, 2).sum()))


    def train(self, loss, optimizer, x_val, y_val,session_ids,
              l1_loss_weight = 0,  # when zero, no L1 regularization
              l2_loss_weight = 0,
              gpu=False):
        """
            Train the model with a batch (in our case, also a session) of data
        """
        #add session_id
        # expect y_val to be of one_dimension
        y_val = y_val.reshape(len(y_val), 1)

        tensorX = torch.from_numpy(x_val).double()
        tensorY = torch.from_numpy(y_val).double()

        if (gpu):
            dtype = torch.cuda.DoubleTensor
        else:
            dtype = torch.DoubleTensor

        # input variable
        x = Variable(tensorX.type(dtype), requires_grad=False)
        # target variable
        y = Variable(tensorY.type(dtype), requires_grad=False)

        # Forward to calculate the losses
        ids = torch.from_numpy(session_ids)
        fx = self.forward(x,ids)

        # Calculate losses
        data_loss = loss.forward(fx, y)

        # optional: add L1 or L2 penalities for regularization
        if (l1_loss_weight != 0):
            l1_loss = self.l1_loss(l1_loss_weight)
            output = data_loss + l1_loss

        elif (l2_loss_weight != 0):
            l2_loss = self.l2_loss(l2_loss_weight)
            output = data_loss + l2_loss

        else:
            output = data_loss

        # Underflow in the loss calculation
        if math.isnan(output.data.item()):
            raise ValueError('NaN detected')
            #return output.item()

        if (isinstance(optimizer, torch.optim.LBFGS)):
            def closure():
                optimizer.zero_grad()
                output.backward(retain_graph=True)
                return output

            optimizer.step(closure)
        else:
            # Reset gradient
            optimizer.zero_grad()
            # Backward
            output.backward()
            # Update parameters
            optimizer.step()

        # return the cost
        return output.data.item()


    def predict(self, x_val, ids, binary=False):
        '''
            Give prediction for alternatives within a single session
            x_val: DataFrame, or np.ndarray
            return: numpy
        '''
        is_gpu = self.get_params()[0].is_cuda

        ids = torch.from_numpy(ids)

        if isinstance(x_val, pd.DataFrame):
            tensorX = torch.from_numpy(x_val.values).double()
        else:
            tensorX = torch.from_numpy(x_val).double()

        if (is_gpu):
            x = Variable(tensorX.type(torch.cuda.DoubleTensor), requires_grad=False)
        else:
            x = Variable(tensorX, requires_grad=False)

        output = self.forward(x,ids)

        if (is_gpu):
            # get the data from the memory of GPU into CPU
            prob = output.cpu().data.numpy()
        else:
            prob = output.data.numpy()

        if (binary):
            # convert the softmax values to binary values
            max_indice = prob.argmax(axis=0)
            ret = np.zeros(len(prob))
            ret[max_indice] = 1
            return ret
        else:
            return prob


    def get_params(self):
        '''
            Return the Variable of the MNL parameters,
              which can be updated manually.
        '''
        for name, param in self.named_parameters():
            if param.requires_grad and name == 'linear.weight':
                return param
        return None


    def print_params(self):
        '''
            Print all the parameters within the model
        '''
        params = self.get_params()[0]

        if (params.is_cuda):
            values = params.cpu().data.numpy()
        else:
            values = params.data.numpy()

        for index, feature in enumerate(self.feature_list):
            print(feature, ':', values[index])


    def get_feature_weight(self, feature_name):
        ''' Retrieve the weight of the desired feature '''
        params = self.get_params()[0]

        if (params.is_cuda):
            param_values = params.cpu().data.numpy()
        else:
            param_values = params.data.numpy()

        for index, feature in enumerate(self.feature_list):
            if (feature_name == feature):
                return param_values[index]

        # did not find the specified feature
        return None


    def get_feature_weights(self):
        ''' get the dictionary of feature weights '''
        params = self.get_params()[0]

        if (params.is_cuda):
            param_values = params.cpu().data.numpy()
        else:
            param_values = params.data.numpy()

        feature_weights = {}

        for index, feature in enumerate(self.feature_list):
            feature_weights[feature] = param_values[index]

        return feature_weights


    def set_feature_weight(self, feature_name, value):
        ''' Reset the specified feature weight
        '''
        params = self.get_params()[0]
        is_found = False

        try:
            for index, feature in enumerate(self.feature_list):
                if (feature_name == feature):
                    is_found = True
                    # override the parameters within the model
                    params[index] = value
        except RuntimeError as e:
            #print('RuntimeError: ', e)
            #print('One can ignore this error, since the parameters are still updated !')
            pass

        return is_found


    def save(self, file_name):
        '''
            Serialize the model object into a file
        '''
        with open(file_name, mode='wb') as model_file:
            pickle.dump(self, model_file)
            print('save model to ', file_name)


    def set_train_config(self, train_config):
        '''
            Set the training configs along with the model,
             so that it can be serialized together with the model.
        '''
        self.train_config = train_config


    def get_train_config(self):
        return self.train_config


def load_model(pickle_file):
    """
        Instantialize a model from its pickle file.
    """
    with open(pickle_file, 'rb') as inp:
        try:
            # python 3
            model = pickle.load(inp, encoding='bytes')
        except:
            # python 2
            model = pickle.load(inp)

        print('load model from ', pickle_file)
        return model


def build_model(input_dim):
    '''
        Another way to build the model.
    '''
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, 1, bias=False))

    # We need the softmax layer here for the binary cross entropy later
    model.add_module('softmax', torch.nn.Softmax(dim=1))

    return model
