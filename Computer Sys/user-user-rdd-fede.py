from pyspark import SparkContext
import csv
#from pyspark.mllib.linalg import Matrix, Matrices
from scipy import linalg, sparse
import numpy as np
from itertools import groupby
from operator import itemgetter

sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("data/train.csv")
#icm_rdd = sc.textFile("data/icm.csv")
test_rdd= sc.textFile("data/test.csv")

train_header = train_rdd.first()
#icm_header = icm_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
#icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))

test_users=test_clean_data.map( lambda x: int(x[0])).collect()
#test_users.take(10)

#for every item all its features
#rouped_features = sc.parallelize([(1,[1,2]),(2,[2,3,4]),(3,[3,4]),(4,[1,2,4])])
#grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1])))
#grouped_features.take(10)
#grouped_features.cache()

#for every features all its items
#grouped_items = sc.parallelize([(1,[1,4]),(2,[1,2,4]),(3,[2,3]),(4,[2,3,4])])
#grouped_items = icm_clean_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
#grouped_items.take(10)
#grouped_items.cache()

#for every user all its ratings (item, rate)
#grouped_rates = sc.parallelize([(1,[(1,8),(3,2)]),(2,[(1,2),(2,9),(3,7)]),(3,[(3,1),(4,10)])])
grouped_rates = train_clean_data.map(lambda x: (x[0],(x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
#grouped_rates.take(10)
grouped_rates.cache()

#return only test users
def is_in_test(user):
    return user[0] in test_users

test_user_ratings = grouped_rates.filter(is_in_test).sortByKey()
#test_user_ratings.take(10)
test_user_ratings.cache()

#for every item all its ratings
item_ratings = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
#item_ratings.take(10)
shrinkage_factor = 20
item_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()
#item_ratings_mean

#for every test user calculates its model
#i = 0
def parse_neighbors(line):
    line_no_simbols = line.replace('"', "").replace("[", "").replace(" ", "").replace("]","")
    elements = line_no_simbols.split(",")

    user = elements.pop(0)

    if elements[0]!='':
        return (int(user),[int(x) for x in elements[:20]])
    return(int(user),[])

user_neighbors_raw = sc.textFile("losKNN20_1.csv")
user_neighbors_clean = user_neighbors_raw.map(parse_neighbors)
collected_user_neighbors_clean=dict(user_neighbors_clean.collect())

f = open('submission2.csv', 'wt')
writer = csv.writer(f)
writer.writerow(('userId','RecommendedItemIds'))
#i = 0
k = 20
shrinkage_factor_knn = 20
for u in test_user_ratings.toLocalIterator():
    KNN = collected_user_neighbors_clean[u[0]]
    already_voted = test_user_ratings.filter(lambda y: u[0] == y[0]).flatMap(lambda x: x[1]).map(lambda x: x[0]).collect()
    items_of_similar_users = grouped_rates.filter(lambda x:x[0] in KNN).flatMap(lambda x: x[1])#.collect()
    items_ratings_similar = items_of_similar_users.aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
    items_ratings_similar_mean = items_ratings_similar.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor_knn))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()
    predictions = items_ratings_similar_mean[:5]
    iterator = 0
    for i in range(5 - len(predictions)):
        while (item_ratings_mean[iterator] in already_voted) or (item_ratings_mean[iterator] in predictions):
            iterator = iterator + 1
        predictions = predictions + [item_ratings_mean[iterator]]
    writer.writerow((u[0], '{0} {1} {2} {3} {4}'.format(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])))
f.close()

#15374 utente max

#37141 item max

#19715 feature max
