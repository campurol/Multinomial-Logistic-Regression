'''

###################################################################################

The MIT License (MIT)

Copyright (c) 2018 Lisong Guo <lisong.guo@me.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

###################################################################################
'''

def train_one_epoch(epoch_index, module_tuple, df_session_groups, train_config):
    '''
    '''
    (model, loss, optimizer) = module_tuple
    
    gpu = train_config['gpu']
    verbose = train_config['verbose']
    l1_loss_weight = train_config['l1_loss_weight']
    l2_loss_weight = train_config['l2_loss_weight']
    MNL_features = train_config['MNL_features']
    save_gradients = train_config.get('save_gradients', False)

    total_cost = 0
    if (verbose >= 2):
        print('Num. sessions:', len(df_session_groups))
    
    for session_id in list(df_session_groups.groups.keys()):
    
        df_session = df_session_groups.get_group(session_id)
    
        if (verbose >= 2):
            print('-----------------------')
            print('session_id:', session_id)
            print('No. alternatives:', len(df_session))
    
    
        cost = train(model, loss, optimizer,
                     df_session[MNL_features].values,
                     df_session['choice'].values,
                     l1_loss_weight = l1_loss_weight,  # when zeor, no regularization
                     l2_loss_weight = l2_loss_weight,  # when zeor, no regularization
                     gpu=gpu)
        
        total_cost += cost
    
        # save the gradients if asked
        if (save_gradients):
            new_gradients = get_session_gradients(epoch_index, session_id, model.parameters())
            train_config['session_gradients'].extend(new_gradients)
        
        if (verbose >= 2):
            print('train cost:', cost)
            predY = model.predict(df_session[MNL_features].values)
            print('Real Y-value:', df_session['choice'].values)
            print('Prediction:', predY)

    return total_cost


def train_with_early_stopping(model_tuple, train_data, train_config):
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
    
    if (save_gradients):
        # a variable that carries over epoches
        train_config['session_gradients'] = []
    
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(epoch, model_tuple, train_data, train_config)
        loss_list.append(epoch_loss)
        
        if (verbose >= 1):
            print('epoch:', epoch, ' loss:', epoch_loss, 'best_loss:', best_loss)

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


def get_session_gradients(epoch_index, session_id, parameters):
    '''
        retrieve the gradient values from the Parameter of gradient
    '''
    res = []
    for param in parameters:
        if (param.is_cuda):
            gradients = param[0].cpu().data.numpy()
        else:
            gradients = param[0].data.numpy()
        
        res.append({
            'epoch_id': epoch_index,
            'session_id': session_id, 
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


def run_training(df_training, train_config, model_tuple=None):
    '''
    '''
    MNL_features = train_config.get('MNL_features', [])
    
    if (len(MNL_features) == 0):
        # use all the applicable features in the data, excluding session specific features
        MNL_features = get_default_MNL_features(df_training)
        
        # set the config for the later use
        train_config['MNL_features'] = MNL_features
    
    n_features = len(MNL_features)
    print('Num features:', n_features)
    print('========================')
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
    df_session_groups = df_training.groupby('session_id')
    
    loss_list = train_with_early_stopping(model_tuple, df_session_groups, train_config)

    return (model_tuple, loss_list)


def plot_loss(loss_list):
    '''
        plot the loss evolution
    '''
    ax = pd.Series(loss_list, name='loss').plot()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    _ = ax.set_title('Loss Evolution during MNL training')
