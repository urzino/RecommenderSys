from pyspark import SparkContext
from scipy import sparse as sm
from sklearn.preprocessing import normalize
import numpy as np
import csv
sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("data/train.csv")
test_rdd= sc.textFile("data/target_users.csv")

train_header = train_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))

test_users=test_clean_data.map( lambda x: int(x[0])).collect()


grouped_rates = train_clean_data.filter(lambda x: x[0] in test_users).map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
grouped_rates_dic = dict(grouped_rates)


item_ratings = train_clean_data.map(lambda x: (x[0], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
user_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1]))).collect()
user_ratings_mean_dic=dict(user_ratings_mean)


item_ratings_forTop = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
#item_ratings.take(10)
shrinkage_factor = 20
item_ratings_mean = item_ratings_forTop.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()


users = train_clean_data.map(lambda x: x[0]).collect()
items = train_clean_data.map(lambda x: x[1]).collect()
ratings = train_clean_data.map(lambda x: x[2]-user_ratings_mean_dic[x[0]]).collect()


UxI= sm.csr_matrix((ratings, (users, items)))



#tipo 1
UxI_norm=sm.csr_matrix(normalize(UxI,axis=0))
IxI_sim=sm.csr_matrix(UxI_norm.T.dot(UxI_norm))
IxI_sim.setdiag(0)
UxI_pred=sm.csr_matrix(UxI.dot(IxI_sim))


#tipo 2

UxI_norm=sm.csr_matrix(normalize(UxI,axis=1))
UxU_sim=sm.csr_matrix(UxI_norm.dot(UxI_norm.T))
UxU_sim.setdiag(0)
UxI_pred=sm.csr_matrix(UxU_sim.dot(UxI))





f = open('submission_collaborative3.csv', 'wt')
writer = csv.writer(f)
writer.writerow(('userId','RecommendedItemIds'))
for user in test_users:
    top=[0,0,0,0,0]

    user_predictions=UxI.getrow(user)
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
