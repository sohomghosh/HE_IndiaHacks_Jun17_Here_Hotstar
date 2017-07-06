import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from gensim.models import word2vec
import logging
import os
import xgboost as xgb

'''
#import json

#data_train_json = json.load(open("/home/sohom/Desktop/HE_IndiaHacks/question-2/train_data.json",'r'))
#train = pd.DataFrame.from_dict(data_train_json, orient='index')

#data_test_json = json.load(open("/home/sohom/Desktop/HE_IndiaHacks/question-2/test_data.json",'r'))
#test = pd.DataFrame.from_dict(data_test_json, orient='index')
'''
train_data = pd.read_json('train_data.json',orient='index')
test_data = pd.read_json('test_data.json',orient='index')

train_data = train_data.replace({'segment':{'pos':1,'neg':0}})
train_data['segment'].value_counts()
train_data['segment'].value_counts()/train_data.shape[0]
#0    0.923725
#1    0.076275
test_data['segment']=float('nan')
train_test = train_data.append(test_data)

genres_list_of_lists=train_test['genres'].apply(lambda x : [i.split(':')[0].lower() for i in str(x).split(',')])
Counter([item for sublist in genres_list_of_lists for item in sublist])
#Counter({'cricket': 169004, 'drama': 148086, 'romance': 106588, 'reality': 86356, 'talkshow': 79421, 'comedy': 70515, 'family': 66469, 'action': 53861, 'livetv': 41857, 'crime': 35119, 'thriller': 34566, 'mythology': 18910, 'awards': 18522, 'football': 11428, 'badminton': 10273, 'horror': 9985, 'sport': 9088, 'kids': 7364, 'travel': 6442, 'wildlife': 5978, 'teen': 3137, 'science': 2757, 'hockey': 2542, 'documentary': 1375, 'kabaddi': 1358, 'formula1': 535, 'tennis': 427, 'athletics': 391, 'table tennis': 323, 'formulae': 45, 'volleyball': 42, 'boxing': 41, 'na': 17, 'swimming': 10, 'indiavssa': 2})
genres=list(set([item for sublist in genres_list_of_lists for item in sublist]))

cities_list_of_lists=train_test['cities'].apply(lambda x : [' '.join(i.split(':')[:-1]).lower() for i in str(x).split(',')])
cities_list_of_lists_times=train_test['cities'].apply(lambda x : [int(i.split(':')[-1]) for i in str(x).split(',')])
Counter([item for sublist in cities_list_of_lists for item in sublist])
cities=list(set([item for sublist in cities_list_of_lists for item in sublist]))

dow_list_of_lists=train_test['dow'].apply(lambda x : [i.split(':')[0].lower() for i in str(x).split(',')])
Counter([item for sublist in dow_list_of_lists for item in sublist])
#Counter({'4': 192073, '7': 187992, '6': 182206, '5': 180716, '3': 174859, '1': 174669, '2': 169375})
dow=list(set([item for sublist in dow_list_of_lists for item in sublist]))

titles_list_of_lists=train_test['titles'].apply(lambda x : [' '.join(i.split(':')[:-1]).lower() for i in str(x).split(',') if i.split(':')[-1]!=''])
titles_list_of_lists_times=train_test['titles'].apply(lambda x : [int(i.split(':')[-1]) for i in str(x).split(',') if i.split(':')[-1]!='']) 
Counter([item for sublist in titles_list_of_lists for item in sublist])
titles=list(set([item for sublist in titles_list_of_lists for item in sublist]))

tod_list_of_lists=train_test['tod'].apply(lambda x : [i.split(':')[0].lower() for i in str(x).split(',')])
Counter([item for sublist in tod_list_of_lists for item in sublist])
#Counter({'21': 159896, '20': 158760, '22': 150989, '19': 150958, '16': 144757, '15': 139572, '18': 139160, '17': 137029, '14': 135274, '13': 127440, '23': 127336, '12': 115732, '11': 110334, '10': 101205, '0': 87527, '9': 84906, '8': 60323, '1': 58465, '7': 45229, '2': 39250, '6': 30262, '3': 26360, '5': 20331, '4': 19394})
tod=list(set([item for sublist in tod_list_of_lists for item in sublist]))

dow_nm=["day"+i for i in dow]
tod_nm=["time"+i for i in tod]

train_test=pd.concat([train_test,pd.DataFrame(columns=dow_nm)])
train_test=pd.concat([train_test,pd.DataFrame(columns=tod_nm)])
train_test=pd.concat([train_test,pd.DataFrame(columns=genres)])

#cnt=1
for i in list(train_test.index):
	#print(train_test.loc[i,:])
	#[(j.split(':')[0],j.split(':')[1]) for j in train_test.loc[i,'genres'].split(',')]
	for j in train_test.loc[i,'genres'].split(','):
		train_test.loc[i,j.split(':')[0].lower()]=j.split(':')[1]
	#gen={a:b for (a,b) in train_test.loc[i,'genres']}
	for j in train_test.loc[i,'dow'].split(','):
		train_test.loc[i,"day"+j.split(':')[0].lower()]=j.split(':')[1] 
	for j in train_test.loc[i,'tod'].split(','):
		train_test.loc[i,"time"+j.split(':')[0].lower()]=j.split(':')[1] 

	#if cnt==200:b
	#	break
	#cnt=cnt+1



del train_test['genres']
del train_test['dow']
del train_test['tod']

train_test.shape()
train_test.to_csv()
# descr
# sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, max_features=4000, sublinear_tf=True)
# sklearn_representation = sklearn_tfidf.fit_transform(cities_all)
# #tfidf=pd.DataFrame(data=sklearn_representation)
# tfidf=pd.DataFrame(sklearn_representation.todense())
# tfidf.to_csv("train_file_tfidf.csv",index=False)
# x_train=tfidf

#cities-> text to vec
#titles-> text to vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences = [['first', 'sentence'], ['second', 'sentence']]

model_cities = word2vec.Word2Vec(cities_list_of_lists, size=40,min_count =1, window=3, workers =-1,sample=1e-5)
model_cities.wv['gurgaon']

features_cities = np.zeros(shape=(0,40))
for (i,numi) in zip(cities_list_of_lists,cities_list_of_lists_times):
	su=np.zeros(shape=(40))
	for (j,numj) in zip(i,numi):#j is a city
		k=np.array(model_cities.wv[j])*numj
		#print(k)
		su=su+k
		#print(su)
	features_cities=np.vstack([features_cities, su])


np.savetxt("features_cities.csv", features_cities, delimiter=",")

features_titles = np.zeros(shape=(0,40))
model_titles=word2vec.Word2Vec(titles_list_of_lists, size=40,min_count =1, window=3, workers =-1,sample=1e-5)

for i in titles_list_of_lists:
	su=np.zeros(shape=(40))
	for j in i:#j is a title
		k=np.array(model_titles.wv[j])
		su=su+k
	#print(su)
	features_titles=np.vstack([features_titles, su])

np.savetxt("features_titles.csv",features_titles,delimiter=",")

##SOHOM form here
#train_test=pd.read_csv("train_test_v2.csv")

train_test=pd.concat([train_test,pd.DataFrame(features_cities,columns=["city_vec"+str(jh) for jh in list(range(1,41))])],axis=1)
train_test=pd.concat([train_test,pd.DataFrame(features_titles,columns=["title_vec"+str(jh) for jh in list(range(1,41))])],axis=1)

train_test.to_csv("train_test_v4.csv",index=False)

del train_test['cities']
del train_test['titles']

train_test.to_csv("train_test.csv",index=False)

#################################################Deal from here###########################################################
train_test=pd.read_csv("train_test.csv")

features = np.setdiff1d(train_test.columns, ['segment'])

params = {"objective": "binary:logistic","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "seed": 2016, "tree_method": "exact"}

#X_train=train_test[0:len(train_data.index)]
X_train=train_test[0:200000]
#X_test=train_test[len(train_data.index):len(train_test.index)]
X_test=train_test[200000:300000]


dtrain = xgb.DMatrix(X_train[features], X_train['segment'], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds = bst.predict(dtest)

#open("test_ids.txt",'w').write(str(list(test_data.index)))

test_ids=eval(open("test_ids.txt").readline())
submit = pd.DataFrame({'ID': test_ids, 'segment': test_preds})
submit.to_csv("submission_2.csv", index=False)

