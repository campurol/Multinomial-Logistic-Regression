
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
import math

from MNL import *

import itertools

'''
This module provides a number of auxiliary functions, in addition to the MNL model.

- One can find another loss function called MaxLogLikelihoodLoss.

- There is a training function with early stopping capability.

- There are some functions to calculate the KPIs for model benchmarking.

'''


class MaxLogLikelihoodLoss(torch.autograd.Function):
    '''
       the negative of the log likelihood of the chosen alternative.
       Note: this loss function ignores the loss of non-chosen alternatives,
         unlike the BinaryCrossEntropy loss which takes all losses into account.

       But while we maximize the log probability of the chosen alternative,
         we are also minimizing the log probability of the non-chosen ones,
         since we do a softmax over the alternatives within a session.
    '''
    import torch

    def forward(self, input, target):
        # return the negative of the log likelihood of the chosen alternative
        likelihood = torch.dot(torch.t(input).view(-1), target.view(-1))

        # shift the value to the zone [1, 2] to avoid the underflowing
        likelihood = likelihood + 1

        # average over the number of samples
        n_samples = target.size()[0]
        #return torch.neg(torch.log(likelihood) / n_samples)
        return torch.neg(torch.log(likelihood))

def init_model(train_config):
    '''
        build and initialize the MNL model
    '''
    # use the full float type, float64
    torch.set_default_tensor_type('torch.DoubleTensor')

    MNL_features = train_config['MNL_features']
    optimizer_method = train_config['optimizer']
    learning_rate = train_config['learning_rate']
    momentum = train_config['momentum']
    weight_decay = train_config.get('weight_decay', 0)
    loss_func = train_config.get('loss', 'BinaryCrossEntropy')

    #model = build_model(n_features)
    model = MNL(MNL_features)

    # binary cross entropy
    if (loss_func == 'BinaryCrossEntropy'):
        # doc: http://pytorch.org/docs/master/nn.html
        # loss(o,t)=−1/n∑i(t[i]∗log(o[i])+(1−t[i])∗log(1−o[i]))
        loss = torch.nn.BCELoss()
    elif (loss_func == 'MaxLogLikelihood'):
        loss = MaxLogLikelihoodLoss()
    elif (loss_func == 'CrossEntropyLoss'):
        loss = torch.nn.CrossEntropyLoss(size_average=True)

    if (optimizer_method == 'SGD'):
        # e.g. lr = 0.01/ 1e-2
        optimizer = optim.SGD(model.parameters(
        ), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    elif (optimizer_method == 'Adam'):
        # weight_decay:  add L2 regularization to the weights ?
        # It seems that with MNL any regularization would deteriarate the performance.
        # The Adam optimizer seems to converge faster than SGD
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif (optimizer_method == 'Adagrad'):
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif (optimizer_method == 'RMSprop'):
        optimizer = torch.optim.RMSprop(model.parameters(
        ), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    elif (optimizer_method == 'LBFGS'):
        # http://cs231n.github.io/neural-networks-3/#sgd
        # TODO: Even we eliminate the memory concerns, a large downside of naive application of L-BFGS is
        #   that it MUST be computed over the entire training set, which could contain millions of examples.
        #   Unlike mini-batch SGD, getting L-BFGS to work on mini-batches is more tricky and an active area
        #   of research.
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)

    return (model, loss, optimizer)

def get_expanded_data(tobeexpanded, train_config, alter_data, session_data):

    verbose = train_config['verbose']
    alter_features = train_config['alter_features']
    session_features = train_config['session_features']
    alternatives = train_config['alternatives']
    choice_groups = train_config.get(
        'choice_groups', [])  # city_ope and NAICS

    if (verbose >= 2):
        print(f'Working on session {session_id}')
    if (verbose >= 3):
        print(tobeexpanded)

    tobeexpanded['all_altern'] = tobeexpanded.apply(lambda row: alternatives[repr(row[choice_groups].values.tolist())], axis = 1)
    df_session = tobeexpanded.explode('all_altern')
    df_session = df_session.reset_index()

    #create choice variable
    df_session.loc[df_session.alter_id == df_session.all_altern, 'choice'] = 1
    df_session.loc[df_session.alter_id != df_session.all_altern, 'choice'] = 0
    df_session = df_session.drop(columns=['alter_id'])
    df_session = df_session.rename(columns={'all_altern':'alter_id'})

    #fillout missing values for the expanded dataset
    for var in choice_groups:
        df_session[var] = df_session.groupby('session_id')[var].transform('first')

    # install alter_id characteristics (location)
    merge_vars = choice_groups + ['alter_id']
    df_session = df_session.merge(
        alter_data, left_on=merge_vars, right_on=merge_vars, how='left')

    # install session_id characteristics (owner and/or firm)
    df_session = df_session.merge(
            session_data, left_on='session_id', right_on='session_id', how='left')

    # install session-alternative variables
    if len(train_config['session_alter_fn'])!= 0:
        for function in train_config['session_alter_fn']:
            df_session[function] = train_config[function](df_session, train_config)

    df_session = df_session.drop(columns=train_config['drop_vars'])
    return df_session

def train_one_epoch(*, epoch_index, module_tuple, df_sessions, alter_data=None, session_data=None, train_config):
    '''
    '''
    (model, loss, optimizer) = module_tuple

    gpu = train_config['gpu']
    verbose = train_config['verbose']
    l1_loss_weight = train_config['l1_loss_weight']
    l2_loss_weight = train_config['l2_loss_weight']
    MNL_features = train_config['MNL_features']
    save_gradients = train_config.get('save_gradients', False)
    expand = train_config.get('expand', False)

    if (expand):
        alter_features = train_config['alter_features']
        session_features = train_config['session_features']
        alternatives = train_config['alternatives']
        choice_groups = train_config.get(
            'choice_groups', [])  # city_ope and NAICS

    total_cost = 0
    df_session_groups = df_sessions.groupby('session_id')
    if (verbose >= 2):
        print('Num. sessions:', len(df_session_groups))

    datalist=list(df_session_groups.groups.keys())
    train_loader = torch.utils.data.DataLoader(dataset=datalist, batch_size=train_config['batch_size'], shuffle=False)
    for batch_id, batchlist in enumerate(train_loader):
        if (expand):
            #batchlist=batchlist.tolist()
            keep=df_sessions['session_id'].isin(batchlist)
            tobeexpanded = df_sessions.loc[keep].set_index(['session_id', 'alter_id'])
            df_session = get_expanded_data(tobeexpanded, train_config, alter_data, session_data)
            print(f'Batch_id: {batch_id}/{len(train_loader)}: Size={len(tobeexpanded)} expanded to {len(df_session)}')
            #print(df_session.head())
            #print(df_session.info())
        else:
            #batchlist=batchlist.tolist()
            keep=df_sessions['session_id'].isin(batchlist)
            df_session = df_sessions.loc[keep].reset_index()
            print(f'Batch_id: {batch_id}/{len(train_loader)}: Size={len(df_session)}')

        if (verbose >= 2):
            print('-----------------------')
            print('batch_id:', batch_id)
            print('No. Sessions:', len(df_session.session_id.unique()))
            print('No. alternatives:', len(df_session.alter_id.unique()))
            print('No. Observations:', len(df_session))

        try:
            cost = model.train(loss, optimizer,
                               df_session[MNL_features].values,
                               df_session['choice'].values,
                               pd.factorize(df_session['session_id'])[0],
                               l1_loss_weight=l1_loss_weight,  # when zero, no regularization
                               l2_loss_weight=l2_loss_weight,  # when zero, no regularization
                               gpu=gpu)

        except ValueError:
            if (verbose >= 1):
                print('loss underflow in batch: ', batch_id)
            # skip this session
            continue

        total_cost += cost

        # save the gradients if asked
        if (save_gradients):
            new_gradients = get_session_gradients(
                epoch_index, batch_id, model.get_params())
            train_config['session_gradients'].extend(new_gradients)

        if (verbose >= 3):
            print('train cost:', cost)
            predY = model.predict(df_session[MNL_features].values,pd.factorize(df_session['session_id'])[0])
            print('Real Y-value:', df_session['choice'].values)
            print('Prediction:', predY)

    return total_cost


def train_with_early_stopping(*, model_tuple, train_data, alter_data=None, session_data=None, train_config):
    '''
    '''
    wait = 0
    best_loss = 1e15

    loss_list = []

    verbose = train_config['verbose']
    epochs = train_config['epochs']
    patience = train_config['patience']
    early_stop_min_delta = train_config['early_stop_min_delta']
    save_gradients = train_config['save_gradients']
    expand = train_config['expand']

    if (save_gradients):
        # a variable that carries over epoches
        train_config['session_gradients'] = []

    for epoch in range(epochs):
        if (expand):
            epoch_loss = train_one_epoch(epoch_index=epoch, module_tuple=model_tuple, df_sessions=train_data, train_config=train_config,
                                         alter_data=alter_data, session_data=session_data)
        else:
            epoch_loss = train_one_epoch(
                epoch_index=epoch, module_tuple=model_tuple, df_sessions=train_data, train_config=train_config)
        loss_list.append(epoch_loss)

        if (verbose >= 1):
            print('epoch:', epoch, ' loss:',
                  epoch_loss, 'best_loss:', best_loss)

        if (epoch_loss - best_loss) < -early_stop_min_delta:
            # find the new minimal point, reset the clock
            best_loss = epoch_loss
            wait = 1
        else:
            if (wait >= patience):
                print('Early stopping!', ' epoch:', epoch,
                      'min_delta:', early_stop_min_delta, ' patience:', patience)
                break
            wait += 1

    print('Final epoch:', epoch, ' loss:', epoch_loss)

    return loss_list


def get_session_gradients(epoch_index, batch_id, parameters):
    '''
        retrieve the gradient values from the Parameter of gradient
    '''
    parameters = parameters.grad
    res = []
    for param in parameters:
        if (param.is_cuda):
            gradients = param.cpu().data.numpy()
        else:
            gradients = param.data.numpy()

        res.append({
            'epoch_id': epoch_index,
            'batch_id': batch_id,
            'mean_abs_gradients': np.mean(np.abs(gradients)),
            'std_abs_gradients': np.std(np.abs(gradients)),
            'gradients': gradients})

    return res


def get_default_MNL_features(df_data):
    '''
        Retrieve all features from the dataframe,
          excluding the auxliary features.
    '''
    # use all the applicable features in the data, excluding session specific features
    return sorted(set(df_data.columns.values) -
                  set(['session_id', 'alter_id', 'choice']))

def get_alternatives(df_training, train_config):
    alternatives = {}
    choice_groups = train_config.get('choice_groups', [])
    unique_values = df_training[choice_groups].drop_duplicates()

    for groupcombo in unique_values.index:
        cond = ''
        for var in choice_groups:
            equalto = unique_values.loc[groupcombo][var]
            if type(equalto)==str:
                equalto = '"' + equalto + '"'
            if cond == '':
                cond = cond + var + '==' + str(equalto)
            else:
                cond = cond + ' & ' + var + '==' + str(equalto)
        # this assumes that all neighborhoods with no firm are not part of the choice set
        alternatives[repr(df_training.loc[groupcombo,choice_groups].values.tolist())] = df_training.query(cond).alter_id.unique()

    return alternatives

def run_training(*, df_training, train_config, alter_data=None, session_data=None, model_tuple=None):
    '''
    '''
    verbose = train_config.get('verbose', [])
    expand = train_config.get('expand', [])
    alter_features = train_config.get('alter_features', [])
    session_features = train_config.get('session_features', [])
    session_alter_features = train_config.get('session_alter_features', [])
    choice_groups = train_config.get('choice_groups', [])
    extra_choice_feature = train_config.get('extra_choice_feature',[])
    drop_vars = train_config.get('drop_vars',[])
    MNL_features = alter_features + session_features + session_alter_features
    MNL_features = [x for x in MNL_features if x not in drop_vars]
    train_config['MNL_features'] = MNL_features

    if (expand):
        train_config['alternatives'] = get_alternatives(df_training, train_config)

    if (len(MNL_features) == 0 & expand == False):
        # use all the applicable features in the data, excluding session specific features
        MNL_features = get_default_MNL_features(df_training)
        # set the config for the later use
        train_config['MNL_features'] = MNL_features
    elif (len(MNL_features) == 0 & expand == True):
        print('Error, expansion defined but no features')

    n_features = len(MNL_features)
    print('Num features:', n_features)
    print('========================')
    print('All features:', MNL_features)
    print('========================')
    print('Alternative Features:', [x for x in alter_features if x not in drop_vars]  )
    print('========================')
    print('Session Features:', [x for x in session_features if x not in drop_vars]  )
    print('========================')
    print('Loaded but dropped features:', drop_vars)
    print('========================')
    print('Group IDs:', choice_groups)
    print('========================')

    if (verbose>=2):
        print(train_config)
        print('========================')

    if (model_tuple is None):
        # Create a new model, other continue training on the existing model.
        (model, loss, optimizer) = init_model(train_config)

        if (train_config['gpu']):
            # run the model in GPU
            model = model.cuda()

            #hook = model.get_params().register_hook(lambda grad: print(grad))

        model_tuple = (model, loss, optimizer)
    else:
        print('Continue training...')

    # train with early stopping
    if (expand):
        loss_list = train_with_early_stopping(model_tuple=model_tuple, train_data=df_training, train_config=train_config,
                                              alter_data=alter_data, session_data=session_data)
    else:
        loss_list = train_with_early_stopping(
            model_tuple=model_tuple, train_data=df_training, train_config=train_config)

    return (model_tuple, loss_list)


def test_model(model, df_testing, train_config, alter_data, session_data, features_to_skip=None):
    '''
        Test the model with the given data
        train_config: some parameters are used, i.e. MNL_feature, gpu
        features_to_skip:  a list of features to skip in the validation
        return: the statistic results of testing
    '''
    df_session_groups = df_testing.groupby('session_id')

    if (train_config['verbose']):
        print('Num of testing sessions:', len(df_session_groups))

    MNL_features = train_config['MNL_features']
    expand = train_config['expand']

    # the testing data with the prediction value for each alternative
    ret = []

    session_list = list(df_session_groups.groups.keys())

    df_testing_groups = df_testing.groupby('session_id')
    datalist=list(df_testing_groups.groups.keys())

    # Important for the "stochastic" probability of the gradient descent algorithm ?!
    if (train_config.get('shuffle_batch', True)):
        import random
        random.shuffle(datalist)

    train_loader = torch.utils.data.DataLoader(dataset=datalist, batch_size=train_config['batch_size'], shuffle=False)
    for batch_id, batchlist in enumerate(train_loader):
        if (expand):
            #batchlist=batchlist.tolist()
            keep=df_testing['session_id'].isin(batchlist)
            tobeexpanded = df_testing.loc[keep].set_index(['session_id', 'alter_id'])
            df_session = get_expanded_data(tobeexpanded, train_config, alter_data, session_data)
            print(f'Batch_id: {batch_id}/{len(train_loader)}: Size={len(tobeexpanded)} expanded to {len(df_session)}')
            #print(df_session.head())
            #print(df_session.info())
        else:
            #batchlist=batchlist.tolist()
            keep=df_testing['session_id'].isin(batchlist)
            df_session = df_testing.loc[keep].reset_index()
            print(f'Batch_id: {batch_id}/{len(train_loader)}: Size={len(df_session)}')

        df_session_groups = df_session.groupby('session_id')
        for session_id in list(df_session_groups.groups.keys()):
            df_one_session = df_session_groups.get_group(session_id).copy()

            if (features_to_skip == None):
                testing_data = df_one_session[MNL_features].copy()
            else:
                # Set the values of feature-to-skip to be zero,
                #   i.e. nullify the weights associated with the features to skip
                testing_data = df_one_session[MNL_features].copy()
                testing_data[features_to_skip] = 0

            predY = model.predict(testing_data.values, pd.factorize(df_one_session['session_id'])[0], binary=False)

            # add the prediction column
            df_one_session['pred_value'] = predY

            ret.append(df_one_session)

    # concatenate the dataframes along the rows
    return pd.concat(ret, axis=0)


def rank(pred_values, real_choice):
    '''
        Get the rank of chosen alternative within the predicted values
    '''
    # first, rank all the values
    rank = pd.DataFrame(pred_values).rank(axis=0, ascending=False)

    # filter out the rank of the chosen alternative, by dot product
    return rank[0].dot(real_choice)


def mean_rank(pred_values, real_choice):
    '''
        In a session with multiple choices,
         get the mean rank values for the chosen choices.
    '''
    return rank(pred_values, real_choice) / real_choice.sum()


def get_chosen_pred_value(pred_values, real_choice):
    return pd.Series(pred_values.reshape(-1)).dot(real_choice)


def mean_chosen_pred_value(pred_values, real_choice):
    '''
        In a session with multiple choices,
          get the mean probability value for the chosen ones.
    '''
    return get_chosen_pred_value(pred_values, real_choice) / real_choice.sum()


def validate(model, df_testing, train_config, alter_data=None, session_data=None, features_to_skip=None):
    '''
        Test the model with the given data
        train_config: some parameters are used, i.e. MNL_feature, gpu
        features_to_skip:  a list of features to skip in the validation
        return: the statistic results of testing
    '''
    df_session_groups = df_testing.groupby('session_id')

    expand = train_config['expand']
    if (train_config['verbose']):
        print('Num of testing sessions:', len(df_session_groups))

    MNL_features = train_config['MNL_features']
    if (len(MNL_features) == 0):
        MNL_features = get_default_MNL_features(df_testing)


    session_batch = []
    session_batch_size = []
    session_batch_expanded_size = []
    session_size = []
    session_num_chosen_choices = []  # the number of chosen choices
    # rank of chosen one
    session_rank = []
    # the chosen probability
    session_pred_value = []
    # the maximum probability that is assigned to an alternative within a session.
    session_max_prob = []
    session_second_prob = []
    session_third_prob = []
    session_mean_pred_value = []
    session_mean_rank = []
    session_mean_distance = []

    # the second maximum probability
    # the probability of opening at home
    if 'athome' in MNL_features:
        home_rank = []
        home_mean_rank = []
        session_home_value = []
        session_mean_home_value = []
        session_num_chosen_homes = []
        
    df_testing_groups = df_testing.groupby('session_id')
    datalist=list(df_testing_groups.groups.keys())
    # Important for the "stochastic" probability of the gradient descent algorithm ?!
    if (train_config.get('shuffle_batch', True)):
        import random
        random.shuffle(datalist)

    train_loader = torch.utils.data.DataLoader(dataset=datalist, batch_size=train_config['batch_size'], shuffle=False)
    for batch_id, batchlist in enumerate(train_loader):
        if (expand):
            #batchlist=batchlist.tolist()
            keep=df_testing['session_id'].isin(batchlist)
            tobeexpanded = df_testing.loc[keep].set_index(['session_id', 'alter_id'])
            df_session = get_expanded_data(tobeexpanded, train_config, alter_data, session_data)
            print(f'Batch_id: {batch_id}/{len(train_loader)}: Size={len(tobeexpanded)} expanded to {len(df_session)}')
            #print(df_session.head())
        else:
            #batchlist=batchlist.tolist()
            keep=df_testing['session_id'].isin(batchlist)
            df_session = df_testing.loc[keep].reset_index()
            print(f'Batch_id: {batch_id}/{len(train_loader)}: Size={len(df_session)}')

        df_session_groups = df_session.groupby('session_id')
        for session_id in list(df_session_groups.groups.keys()):
            df_one_session = df_session_groups.get_group(session_id).copy()

            if (features_to_skip == None):
                testing_data = df_one_session[MNL_features].copy()
            else:
                # Set the values of feature-to-skip to be zero,
                #   i.e. nullify the weights associated with the features to skip
                testing_data = df_one_session[MNL_features].copy()
                testing_data[features_to_skip] = 0

            predY = model.predict(testing_data.values, pd.factorize(df_one_session['session_id'])[0], binary=False)

            #print('SessionId:', session_id)
            #print('AlterId:', df_session['alter_id'].values)
            #print('Real Y-value:', df_session['choice'].values)
            #print('Prediction:', type(predY))


            choice_value = df_one_session['choice'].values
            session_num_chosen_choices.append(choice_value.sum())
            session_size.append(len(df_one_session))
            session_pred_value.append(get_chosen_pred_value(predY, choice_value))
            session_rank.append(rank(predY, choice_value))
            session_mean_pred_value.append(mean_chosen_pred_value(predY, choice_value))
            session_mean_rank.append(mean_rank(predY, choice_value))
            predYlist = predY.T.tolist()[0]
            session_max_prob.append(np.max(predYlist))
            #this will be repeated within a batch_id
            session_batch.append(batch_id)
            session_batch_size.append(len(batchlist))
            session_batch_expanded_size.append(len(df_session))
            session_mean_distance.append(np.mean(df_one_session['distance'].values))
            if 'athome' in MNL_features:
                home_value   = df_one_session['athome'].values                        #probability of openning at home
                session_home_value.append(get_chosen_pred_value(predY, home_value))
                session_mean_home_value.append(mean_chosen_pred_value(predY, home_value))
                home_rank.append(rank(predY, home_value))
                home_mean_rank.append(mean_rank(predY, home_value))
                session_num_chosen_homes.append(home_value.sum())
            
            if len(predYlist)<0:                    ##FIX THIS AT SOME POINT
                nosecondmax = predYlist
                nosecondmax = nosecondmax.remove(np.max(nosecondmax))
                nothirdmax = nosecondmax
                nothirdmax = nothirdmax.remove(np.max(nothirdmax))
                session_second_prob.append(np.max(nosecondmax)) ##second max probability
                session_third_prob.append(np.max(nothirdmax)) ##third max probabiity,
            else:
                session_second_prob.append(np.max(predYlist)) ##second max probability
                session_third_prob.append(np.max(predYlist)) ##third max probabiity,
            

    df_session_KPIs = pd.DataFrame()
    df_session_KPIs['session_id'] = list(df_testing_groups.groups.keys())
    df_session_KPIs['session_size'] = session_size
    df_session_KPIs['num_chosen_choices'] = session_num_chosen_choices
    df_session_KPIs['rank_of_chosen_one'] = session_rank
    df_session_KPIs['prob_of_chosen_one'] = session_pred_value
    df_session_KPIs['mean_rank_of_chosen_one'] = session_mean_rank
    df_session_KPIs['mean_prob_of_chosen_one'] = session_mean_pred_value    
    df_session_KPIs['max_prob'] = session_max_prob
    df_session_KPIs['second_highest_prob'] = session_second_prob
    df_session_KPIs['third_highest_prob'] = session_third_prob
    df_session_KPIs['session_mean_distance'] = session_mean_distance
    if 'athome' in MNL_features:
        df_session_KPIs['prob_of_home'] = session_home_value
        df_session_KPIs['rank_of_home'] = home_rank
        df_session_KPIs['mean_prob_of_home'] = session_mean_home_value
        df_session_KPIs['mean_rank_of_home'] = home_mean_rank        
        df_session_KPIs['num_chosen_homes'] = session_num_chosen_homes

    return df_session_KPIs


def summarize_KPIs(df_session_KPIs, n_features):

    from scipy import stats

    KPI_summary = {}

    #stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='weak')
    # expected 80.0
    KPI_summary['session_num'] = len(df_session_KPIs)
    KPI_summary['mean_session_size'] = df_session_KPIs['session_size'].mean()

    # calculate the percentile, LESS THAN and EQUAL to the given score
    KPI_summary['top_1_rank_quantile'] = \
        stats.percentileofscore(
            df_session_KPIs['rank_of_chosen_one'], 1, kind='weak')

    KPI_summary['top_5_rank_quantile'] = \
        stats.percentileofscore(
            df_session_KPIs['rank_of_chosen_one'], 5, kind='weak')

    KPI_summary['top_10_rank_quantile'] = \
        stats.percentileofscore(
            df_session_KPIs['rank_of_chosen_one'], 10, kind='weak')

    # The ratio between the rank of the chosen alternative and the number of alternatives
    rank_ratio = (
        df_session_KPIs['rank_of_chosen_one'] / df_session_KPIs['session_size'])
    KPI_summary['mean_rank_ratio'] = rank_ratio.mean()
    KPI_summary['median_rank_ratio'] = rank_ratio.median()

    KPI_summary['mean_rank'] = df_session_KPIs['rank_of_chosen_one'].mean()
    KPI_summary['median_rank'] = df_session_KPIs['rank_of_chosen_one'].median()

    KPI_summary['mean_probability'] = df_session_KPIs['prob_of_chosen_one'].mean()
    KPI_summary['median_probability'] = df_session_KPIs['prob_of_chosen_one'].median()

    # the difference of probability values between the chosen one and the predicted one
    prob_diff = (
        df_session_KPIs['prob_of_chosen_one'] - df_session_KPIs['max_prob'])
    KPI_summary['mean_probability_diff'] = prob_diff.mean()
    KPI_summary['median_probability_diff'] = prob_diff.median()

    # the log likelihood for each chosen alternative is negative. The higher the probability,
    #  the closer the log likelihood is to the zero, (i.e. the lower the absolute value)
    KPI_summary['log_likelihood'] = np.log(
        df_session_KPIs['prob_of_chosen_one']).sum()

    KPI_summary['mean_log_likelihood'] = np.log(
        df_session_KPIs['prob_of_chosen_one']).mean()

    # AIC <- 2*length(model$coefficients) - 2*model$loglikelihood
    # Akaike Information Criterion, which estimates the quality of the model, (i.e. the lower, the better)
    '''
      AIC is founded on information theory: it offers an estimate of the relative information lost
        when a given model is used to represent the process that generated the data.
        (In doing so, it deals with the trade-off between the goodness of fit of the model and
         the simplicity of the model.)
    '''
    KPI_summary['AIC'] = 2 * n_features - 2 * KPI_summary['log_likelihood']

    return KPI_summary


def plot_loss(loss_list):
    '''
        plot the loss evolution
    '''
    import pandas as pd
    ax = pd.Series(loss_list, name='loss').plot()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    _ = ax.set_title('Loss Evolution during MNL training')
