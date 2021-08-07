#for distance function
from math import sin, cos, sqrt, atan2, radians

#general packages
import pandas as pd

#for weighted sum function
import numpy as np
#define choice-alternative function

def distance(data, train_config):

	lon1, lat1, lon2, lat2 = data['lat_ope'],data['lon_ope'],data['neigh_lat'],data['neigh_lon']
	
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
	
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	
	a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
	
	c = 2 * np.arcsin(np.sqrt(a))
	
	km = 6367 * c
	
	return km

def athome(data,train_config):
	distance=data['distance'].values
	athome = np.where(distance==0, 1, distance)

	return athome
   
config = {

	'drop_vars':['lat_ope','lon_ope','neigh_lat','neigh_lon'] ,
	# options: BinaryCrossEntropy, MaxLogLikelihood
	#'loss':  'MaxLogLikelihood',
	'loss':  'MaxLogLikelihood',
	
	'expand': True,
	
	'optimizer': 'Adam',  # options:  Adam, RMSprop, SGD, LBFGS.
	# Adam would converge much faster
	# LBFGS is a very memory intensive optimizer (it requires additional param_bytes * (history_size + 1) bytes).
	# If it doesn’t fit in memory try reducing the history size, or use a different algorithm.
	# By default, history_size == 100
	'learning_rate': 0.05, # Applicable to Adam, SGD, and LBFGS
	# The learning_rate parameter seems essential to LBFGS, which converges in two epochs.
	#  So far, learning_rate == 0.1 seems to be ok for LBFGS
	
	#'momentum': 0.9,  # applicable to SGD, RMSprop
	'momentum': 0.01,  # applicable to SGD, RMSprop
	
	# The resulting model seems to be more balanced, i.e. no extreme large/small weights,
	#  although one might not have the most ideal performance, i.e. high top_5_rank etc.
	'weight_decay': 0, # Applicable to Adam, RMSprop and SGD
	

	# indicates the number of sessions included in each batch
	'batch_size': 500,
	
	#maximum number of epochs
	'epochs': 20,
	
	#tolerance for early stopping
	'early_stop_min_delta': 1e-4,
	'patience': 10,
	
	#if able to use GPU (unfortunately I am not able to do this)
	'gpu': False,  # luckily, running on GPU is faster than CPU in this case.
	
	# level of logging, 0: no log,  1: print epoch related logs;  2: print session related logs
	'verbose': 1,
	
	# Adding the regularization degredates the performance of model
	#   which might suggests that the model is still underfitting, not overfitting.
	'l1_loss_weight': 0,  # e.g. 0.001 the regularization that would marginalize the weights
	'l2_loss_weight': 0,
	
	# flag indicates whether to save gradients during the training
	'save_gradients': True, 
	
	#choice alternative function name
	'session_alter_fn': [distance,athome],
	'distance' : distance,
	'athome' : athome	
}
