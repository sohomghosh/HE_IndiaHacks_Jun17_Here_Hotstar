import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import numpy as np

train = pd.read_csv("/home/sohom/Desktop/HE_IndiaHacks/question-1/train.csv")
test = pd.read_csv("/home/sohom/Desktop/HE_IndiaHacks/question-1/test.csv")
train.head()
test.head()
train['DetectedCamera'].value_counts()

#encode as integer
mapping = {'Front':0, 'Right':1, 'Left':2, 'Rear':3}
train = train.replace({'DetectedCamera':mapping})
test = test.replace({'DetectedCamera':mapping})

#renaming column
train.rename(columns = {'SignFacing (Target)': 'Target'}, inplace=True)

#encode Target Variable based on sample submission file
mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
train = train.replace({'Target':mapping})

#target variable
y_train = train['Target']
test_id = test['Id']

#drop columns
train.drop(['Target','Id'], inplace=True, axis=1)
test.drop('Id',inplace=True,axis=1)

#######################################################################################################################################
#################################################RANDOM FOREST MODEL####################################################################
#######################################################################################################################################

#train model
clf = RandomForestClassifier(n_estimators=500,max_features=3,min_samples_split=5,oob_score=True)
clf.fit(train, y_train)

#predict on test data
pred = clf.predict_proba(test)

#write submission file and submit
columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("sub_rfih.csv", index=False)

#######################################################################################################################################
############################################XG-BOOST MODEL#############################################################################
#######################################################################################################################################


dtrain = xgb.DMatrix(train, y_train, missing=np.nan)
dtest = xgb.DMatrix(test, missing=np.nan)

nrounds = 260
watchlist = [(dtrain, 'train')]

params = {"objective": "multi:softprob","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,"num_class":4,
                "seed": 2016, "tree_method": "exact"}


bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds = bst.predict(dtest)

pred=test_preds
columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("sub_xgb.csv", index=False,float_format='%.6f')


#######################################################################################################################################
#################################################USING PARAM GRID######################################################################
#######################################################################################################################################

cv_params = {'max_depth': [2,3], 'min_child_weight': [3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 10, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, "objective": "multi:softprob"}


cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, "objective": "multi:softprob"}

optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1) 
optimized_GBM.fit(train, y_train)

print(optimized_GBM.grid_scores_)
print(optimized_GBM.best_params_)
print(optimized_GBM.best_score_)

#Made by observing the values printed above
best_params = {'learning_rate': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, "objective": "multi:softprob", 'max_depth':3, 'min_child_weight':5,'num_class':4}

final_gb = xgb.train(best_params, dtrain, num_boost_round = 432)

test_preds = final_gb.predict(dtest) # Predict using our testdmat

pred=test_preds
columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv("sub_xgb_param_grid.csv", index=False,float_format='%.6f')


#######################################################################################################################################
gsearch1 = GridSearchCV(cv=5, error_score='raise',estimator=xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,objective='multi:softprob', reg_alpha=0, reg_lambda=1,scale_pos_weight=1, seed=0, silent=True, subsample=0.8),fit_params={}, iid=True, n_jobs=-1,param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7]},pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)

gsearch1.fit(train, y_train)

print(gsearch1.grid_scores_)
#[mean: 0.95796, std: 0.00803, params: {'max_depth': 3, 'min_child_weight': 1}, mean: 0.95793, std: 0.00750, params: {'max_depth': 3, 'min_child_weight': 3}, mean: 0.95770, std: 0.00762, params: {'max_depth': 3, 'min_child_weight': 5}, mean: 0.95570, std: 0.00752, params: {'max_depth': 5, 'min_child_weight': 1}, mean: 0.95609, std: 0.00730, params: {'max_depth': 5, 'min_child_weight': 3}, mean: 0.95635, std: 0.00726, params: {'max_depth': 5, 'min_child_weight': 5}, mean: 0.95492, std: 0.00735, params: {'max_depth': 7, 'min_child_weight': 1}, mean: 0.95455, std: 0.00722, params: {'max_depth': 7, 'min_child_weight': 3}, mean: 0.95502, std: 0.00722, params: {'max_depth': 7, 'min_child_weight': 5}]
print(gsearch1.best_params_)
#{'max_depth': 3, 'min_child_weight': 1}

print(gsearch1.best_score_)
#0.95795764583604

#from sklearn.metrics import accuracy_score
#y_pred = gsearch1.predict(dtest)
#y_pred[y_pred > 0.5] = 1
#y_pred[y_pred <= 0.5] = 0
#accuracy_score(y_pred, y_test), 1-accuracy_score(y_pred, y_test)



#######################################################################################################################################
############################################XG-BOOST PARAMETERS' VALUES################################################################
#######################################################################################################################################
#INPUTS FROM: https://jessesw.com/XG-Boost/
#INPUTS FROM: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
#Source https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions
#HYPERPARAMETER		TUNNING_APPROACH		#RANGE					##NOTE

#Number_of_trees 	Fixed_Value				100-1000				Depending_on_data_size
#Learning_rate    	Fixed_value=>Fine_tune 	[2-10]/Number_of_trees  Depending on number of trees
#Row_sampling		grid_search 			[.5,.75,1.0]
#Column_samling		grid_search 			[.4,.6,.8,1.0]
#Min_leaf_weight 	Fixed_value=>Fine_tune 	3/(% of rare events)	Rule_of_thumb
#Max_Tree_Depth		grid_search 			[4,6,8,10]
#Min_split_gain		Fixed 					0						Keep it zero


