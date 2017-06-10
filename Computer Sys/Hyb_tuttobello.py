
# coding: utf-8

# In[ ]:

from pyspark import SparkContext
from scipy import sparse as sm
from sklearn.preprocessing import normalize
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import spearmanr
from scipy.stats import pearsonr as pears
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm
import time
sc = SparkContext.getOrCreate()


# In[ ]:

train_rdd = sc.textFile("data/train.csv")
icm_rdd = sc.textFile("data/icm_fede.csv")
test_rdd= sc.textFile("data/target_users.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))

test_users=test_clean_data.map( lambda x: int(x[0])).collect()


grouped_rates = train_clean_data.filter(lambda x: x[0] in test_users).map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
grouped_rates_dic = dict(grouped_rates)


item_ratings = train_clean_data.map(lambda x: (x[0], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
user_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1]))).collect()
user_ratings_mean_dic=dict(user_ratings_mean)


item_ratings_forTop = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
#item_ratings.take(10)
shrinkage_factor = 5
item_ratings_mean = item_ratings_forTop.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()


users = train_clean_data.map(lambda x: x[0]).collect()
items = train_clean_data.map(lambda x: x[1]).collect()
ratings = train_clean_data.map(lambda x: x[2]).collect()
ratings_unbiased = train_clean_data.map(lambda x: x[2]-user_ratings_mean_dic[x[0]]).collect()

items_for_features= icm_clean_data.map(lambda x:x[0]).collect()
features = icm_clean_data.map(lambda x:x[1]).collect()
items_for_features.append(37142)
features.append(0)


unos=[1]*len(items_for_features)

UxI= sm.csr_matrix((ratings, (users, items)))
UxI_unbiased= sm.csr_matrix((ratings_unbiased, (users, items)))
IxF= sm.csr_matrix((unos, (items_for_features, features)))


# In[ ]:

n_users,n_items=UxI.shape


# In[ ]:

'''content based shared'''
IxF_normalized=normalize(IxF,axis=1)
NumItems,NumFeatures=IxF.shape
NumFeatures
IDF=[0]*NumFeatures
for i in tqdm(range(NumFeatures)):
    IDF[i]=np.log10(NumItems/len(IxF.getcol(i).nonzero()[1]))
UxF=UxI.dot(IxF_normalized)
FxI=IxF_normalized.multiply(IDF).T
UxI_pred_CB=UxF.dot(FxI).tolil()


# In[ ]:

''' calc items similarity features'''
IxI_sim_f=sm.csr_matrix(cosine_similarity(FxI.T))
IxI_sim_f.setdiag(0)


# In[ ]:

'''calc users similarity features'''
UxU_sim_f=sm.csr_matrix(cosine_similarity(UxF))
UxU_sim_f.setdiag(0)


# In[ ]:

'''collaborative filtering item based + merged similarity'''
#calc similarity
#IxI_sim=UxI_unbiased.T.dot(UxI_unbiased).tocsr()
IxI_sim_dafile=sc.textFile("items-items-sims.csv").map(lambda x: x.replace("(","").replace(")","").replace(" ","").split(",")).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
it1=IxI_sim_dafile.map(lambda x:x[0]).collect()
it2=IxI_sim_dafile.map(lambda x:x[1]).collect()
simsit=IxI_sim_dafile.map(lambda x:x[2]).collect()
IxI_sim=sm.csr_matrix((simsit, (it1, it2)))
IxI_sim.setdiag(0)


# In[ ]:

#merge similarity
IxI_sim= IxI_sim.multiply(2/10) + IxI_sim_f.multiply(8/10)


# In[ ]:

#take knn items
IxI_sim_knn=sm.lil_matrix((n_items,n_items))
k=200
for i in tqdm(range(n_items)):    
    top_k_idx =IxI_sim.getrow(i).toarray()[0].argpartition(-k)[-k:]
    IxI_sim_knn[i,top_k_idx]=IxI_sim[i,top_k_idx]  


# In[ ]:

#calc predictions
UxI_pred_CI=UxI.dot(IxI_sim_knn.T).tolil()


# In[ ]:

'''collaborative filtering user based + merged similarity'''
#calc similarity
#UxU_sim=UxI.dot(UxI.T)
UxU_sim_dafile=sc.textFile("users-users-sims.csv").map(lambda x: x.replace("(","").replace(")","").replace(" ","").split(",")).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
us1=UxU_sim_dafile.map(lambda x:x[0]).collect()
us2=UxU_sim_dafile.map(lambda x:x[1]).collect()
simsus=UxU_sim_dafile.map(lambda x:x[2]).collect()
UxU_sim= sm.csr_matrix((simsus, (us1, us2)))
UxU_sim.setdiag(0)


# In[ ]:

#merge similarity
UxU_sim= UxU_sim.multiply(3/10) + UxU_sim_f.multiply(7/10)


# In[ ]:

#take knn users
UxU_sim_knn=sm.lil_matrix((n_users,n_users))
k=200
for i in tqdm(range(n_users)):    
    top_k_idx =UxU_sim.getrow(i).toarray()[0].argpartition(-k)[-k:]
    UxU_sim_knn[i,top_k_idx]=UxU_sim[i,top_k_idx]  


# In[ ]:

#calc_predictions
UxI_pred_CU=UxU_sim_knn.dot(UxI).tolil()


# In[ ]:

#remove already voted
for user in tqdm(test_users):
    UxI_pred_CB[user,grouped_rates_dic[user]]=0
    UxI_pred_CI[user,grouped_rates_dic[user]]=0
    UxI_pred_CU[user,grouped_rates_dic[user]]=0


# In[ ]:

#rescale algorithms
for user in tqdm(test_users):
    
    row=UxI_pred_CB[user,:].toarray()[0]
    OldMin=min(row)
    OldMax=max(row)
    UxI_pred_CB[user,:]=(((UxI_pred_CB[user,:] - OldMin) * (100 - 0)) / (OldMax - OldMin)) 
    
    row=UxI_pred_CU[user,:].toarray()[0]
    OldMin=min(row)
    OldMax=max(row)
    UxI_pred_CU[user,:]=(((UxI_pred_CU[user,:] - OldMin) * (100 - 0)) / (OldMax - OldMin)) 
    
    row=UxI_pred_CI[user,:].toarray()[0]
    OldMin=min(row)
    OldMax=max(row)
    UxI_pred_CI[user,:]=(((UxI_pred_CI[user,:] - OldMin) * (100 - 0)) / (OldMax - OldMin)) 
    


# In[ ]:

UxI_pred_CB=UxI_pred_CB.tocsr()


# In[ ]:

UxI_pred_CI=UxI_pred_CI.tocsr()


# In[ ]:

UxI_pred_CU=UxI_pred_CU.tocsr()


# In[ ]:

UxI_pred=UxI_pred_CB.multiply(500/1000)+UxI_pred_CI.multiply(300/1000)+UxI_pred_CU.multiply(200/1000)


# In[ ]:

f = open('submission_sum_500-300-200.csv', 'wt')
writer = csv.writer(f)
writer.writerow(('userId','RecommendedItemIds'))

for user in tqdm(test_users):
    top=[0,0,0,0,0]

    user_predictions=UxI_pred.getrow(user)
    iterator = 0
    for i in range(5):
        prediction = user_predictions.argmax()
        while prediction in grouped_rates_dic[user] and prediction != 0:
            user_predictions[0,prediction]=-9
            prediction=user_predictions.argmax()
        if prediction == 0:
            prediction = item_ratings_mean[iterator]
            while prediction in grouped_rates_dic[user] or prediction in top:
                iterator += 1
                prediction = item_ratings_mean[iterator]
            iterator += 1
        else:
            user_predictions[0,prediction]=-9
        top[i]=prediction    
    writer.writerow((user, '{0} {1} {2} {3} {4}'.format(top[0], top[1], top[2], top[3], top[4])))

f.close()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



