capture program drop nmlogit
program define nmlogit
syntax, alter_features(string) ///
				session_features(string) ///
				choice_df(string="default") ///
				indv_id(string="session_id") ///
				alter_id(string="alter_id") ///
				alter_df(string="neigh_df") ///
				session_df(string="indv_df") ///
				expand(string="True")

python: nmlogit( 																			///
						alter_features="`alter_features'",				///
						session_features="`session_features'",		///
						choice_df="`choice_df'",									///
						indv_id="`indv_id'",											///
						alter_id="`alter_id'",										///
						alter_df="`alter_df'",										///
						session_df="`session_df'",								///
						expand="`expand'",												///
					)

end

python:
# import the model and all the auxiliary functions
from MNL import *
from MNL_plus import *
from Mint import *

#
import pandas as pd
import numpy as np
import sfi

def nmlogit(*,alter_features,session_features,choice_df,indv_id, alter_id, alter_df, session_df, expand):
	TRAIN_CONFIG = {
	    #'alter_features': MNL_features,
	    #'session_features'
	    'alter_features':alter_features.split(),
	    'session_features':seassion_features.split(),
	    'choice_groups':choice_groups.split(),

	    # options: BinaryCrossEntropy, MaxLogLikelihood
	    #'loss':  'MaxLogLikelihood',
	    'loss':  'MaxLogLikelihood',

	    'expand': expand,

			'epochs': 20,
	    'early_stop_min_delta': 1e-4,
	    'patience': 5,

			# Adam would converge much faster
	    # LBFGS is a very memory intensive optimizer (it requires additional param_bytes * (history_size + 1) bytes).
	    # If it doesn’t fit in memory try reducing the history size, or use a different algorithm.
	    # By default, history_size == 100
			'optimizer': 'Adam',  # options:  Adam, RMSprop, SGD, LBFGS.

			# The learning_rate parameter seems essential to LBFGS, which converges in two epochs.
			#  So far, learning_rate == 0.1 seems to be ok for LBFGS
			'learning_rate': 0.1, # Applicable to Adam, SGD, and LBFGS

	    #'momentum': 0.9,  # applicable to SGD, RMSprop
	    'momentum': 0,  # applicable to SGD, RMSprop

	    # The resulting model seems to be more balanced, i.e. no extreme large/small weights,
	    #  although one might not have the most ideal performance, i.e. high top_5_rank etc.
	    'weight_decay': 0.01, # Applicable to Adam, RMSprop and SGD

	    'gpu': False,  # luckily, running on GPU is faster than CPU in this case.

	    # level of logging, 0: no log,  1: print epoch related logs;  2: print session related logs
	    'verbose': 1,

	    # Adding the regularization degredates the performance of model
	    #   which might suggests that the model is still underfitting, not overfitting.
	    'l1_loss_weight': 0,  # e.g. 0.001 the regularization that would marginalize the weights
	    'l2_loss_weight': 0,

	    # flag indicates whether to save gradients during the training
	    'save_gradients': True
	}

	#Load from stata
	print ('Loading data')
	alter_connector= sfi.Frame.connect("alter_df")
	session_connector= sfi.Frame.connect("session_df")
	choice_connector= sfi.Frame.connect("choice_df")

	alter_data = alter_connector.get(alter_features)
	session_data = session_connector.get(session_features)
	choices = choice_connector.get(choice_features)

	#Define Pandas Dataframe
	alter_data = pd.DataFrame(data,columns=alter_features)
	session_data = pd.DataFrame(data,columns=session_features)
	choices = pd.DataFrame(data,columns=choice_features)

	#Rename session, alter and choice

	#Run model
	print ('Running model')
	model_tuple, loss_list = run_training(df_training=choices, train_config=TRAIN_CONFIG, alter_data=alter_data, session_data=session_data)
