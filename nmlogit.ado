capture program drop nmlogit
program define nmlogit
syntax [, alter_features(string) 			///
				session_features(string) 	///
				choice_df(string) 			///
				alter_df(string) 			///
				session_df(string) 			///
				indv_id(string) 			///
				alter_id(string) 			///
				expand(string) 				///
				choice_groups(string) 		///
				extra_choice_features(string) ///
				session_alter_features(string) ///
				batch_size(integer 1) ///
				save_model(string)]

if "`choice_df'"==""{
	local choice_df="default"
}
if "`alter_df'"==""{
	local alter_df="neigh_df"
}
if "`session_df'"==""{
	local session_df="indv_df"
}
if "`indv_id'"==""{
	local indv_id="session_id"
}
if "`alter_id'"==""{
	local alter_id="alter_id"
}
if "`expand'"==""{
	local expand="True"
}
				
python: nmlogit( 													///
						alter_features="`alter_features'",			///
						session_features="`session_features'",		///
						choice_df="`choice_df'",					///
						indv_id="`indv_id'",						///
						alter_id="`alter_id'",						///
						alter_df="`alter_df'",						///
						session_df="`session_df'",					///
						expand="`expand'",							///
						batch_size=`batch_size'	,					///
						choice_groups="`choice_groups'"	, ///
						extra_choice_features="`extra_choice_features'", ///
						session_alter_features = "`session_alter_features'", ///
						save_model = "`save_model'" ///
					)
					

ereturn post `b' `V'
ereturn post `se'


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
from nmlogit_config	 import *
	
def nmlogit(*,	alter_features,
				session_features,
				choice_df,
				indv_id,
				alter_id, 
				alter_df, 
				session_df, 
				expand, 
				batch_size, 
				choice_groups, 
				extra_choice_features,
				session_alter_features,
				save_model):
	global config
	
	#import train_config
	config['session_id']=[indv_id]
	config['alter_id']=[alter_id]
	config['alter_features']=alter_features.split()
	config['session_features']=session_features.split()
	config['choice_groups']=choice_groups.split()
	config['session_alter_features']=session_alter_features.split()
	config['extra_choice_features']=extra_choice_features.split()
	config['batch_size']=batch_size
	
	#Load from stata
	print ('Loading data')
	alter_connector= sfi.Frame.connect(alter_df)
	session_connector= sfi.Frame.connect(session_df)
	choice_connector= sfi.Frame.connect(choice_df)

	alter_features   = config['alter_id']   + config['choice_groups'] + config['alter_features']
	session_features = config['session_id'] + config['session_features']
	choice_features  = config['session_id'] + config['alter_id'] + config['choice_groups'] + config['extra_choice_features']
		
	alter_data = alter_connector.get(alter_features)
	session_data = session_connector.get(session_features)
	choices = choice_connector.get(choice_features)
	
	#Define Pandas Dataframe
	alter_data = pd.DataFrame(alter_data,columns=alter_features)
	session_data = pd.DataFrame(session_data,columns=session_features)
	choices = pd.DataFrame(choices,columns=choice_features)

	print(alter_data.head())
	print(session_data.head())
	print(choices.head())
	#Run model
	print ('Running model')
	model_tuple, loss_list = run_training(df_training=choices, train_config=config, alter_data=alter_data, session_data=session_data)
	(model, loss, optimizer) = model_tuple
	
	from scipy.stats import norm
	from operator import truediv

    b=model.get_params()
    gradient=model.get_params().grad
    hessian=torch.mm(torch.t(gradient),gradient)
    se = torch.sqrt(torch.diagonal(hessian))
    t=torch.div(b,se)

    b=b.tolist()
    se=se.tolist()
    gradient=gradient.tolist()
    hessian=hessian.tolist()
    t=t.tolist()

    p=1-norm.cdf(t)
	
	sfi.Matrix.create("b",np.shape(b)[0],np.shape(b)[1],0)
	sfi.Matrix.create("se",np.shape(se)[0],np.shape(se)[1],0)
	sfi.Matrix.create("p",np.shape(p)[0],np.shape(p)[1],0)
	sfi.Matrix.create("V",np.shape(hessian)[0],np.shape(hessian)[1],0)
	sfi.Matrix.create("t",np.shape(t)[0],np.shape(t)[1],0)
    
	sfi.Matrix.store("b",b)
	sfi.Matrix.store("se",se)
	sfi.Matrix.store("p",p)
	sfi.Matrix.store("V",V)
	sfi.Matrix.store("t",t)    
	
	print(f"coefficients: {b} shape {np.shape(b)}")
	print(f"gradient: {gradient}")
	print(f"standard errors: {se} shape {np.shape(se)}")	
	print(f"-values: {p} shape {np.shape(p)}")	
	
	model.save(f'{save_model}.pkl')
	
end