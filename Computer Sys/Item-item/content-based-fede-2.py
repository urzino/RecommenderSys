from pyspark import SparkContext
import csv
from scipy import linalg, sparse
import numpy as np
from itertools import groupby
from operator import itemgetter

sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("../data/train.csv")
icm_rdd = sc.textFile("../data/icm_fede.csv")
test_rdd= sc.textFile("../data/test.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))

test_users=test_clean_data.map( lambda x: int(x[0])).collect()
#test_users=[1,2,3,4]
#test_users.take(10)

#for every item all its features
#rouped_features = sc.parallelize([(1,[1,2]),(2,[2,3,4]),(3,[3,4]),(4,[1,2,4])])
grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1])))
#grouped_features.take(10)
grouped_features.cache()

#for every features all its items
#grouped_items = sc.parallelize([(1,[1,4]),(2,[1,2,4]),(3,[2,3]),(4,[2,3,4])])
grouped_items = icm_clean_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
#grouped_items.take(10)
grouped_items.cache()
grouped_items_dic = dict(grouped_items.collect())

#for every user all its ratings (item, rate)
#grouped_rates = sc.parallelize([(1,[(1,8),(3,2)]),(2,[(1,2),(2,9),(3,7)]),(3,[(3,1),(4,10)])])
grouped_rates = train_clean_data.map(lambda x: (x[0],(x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
#grouped_rates.take(10)
grouped_rates.cache()

#for every item all its ratings
item_ratings = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
#item_ratings.take(10)
shrinkage_factor = 20
item_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()
#.map(lambda x: x[0])
#return only test users
def is_in_test(user):
    return user[0] in test_users

test_user_ratings = grouped_rates.filter(is_in_test).sortByKey()
#test_user_ratings.take(10)
test_user_ratings.cache()

#returns mean of a list using tf/idf for every feature
def mean_ratings_2(rates):
    return sum(rates[1]) * float(len(rates[1])) / len(grouped_items_dic[rates[0]])

#returns mean of a list of ratings for a feature
def mean_ratings(rates):
    return sum(rates) / float(len(rates))

grouped_features_array = grouped_features.collect()

#returns all the features voted by the user
def calculate_features_ratings(user_rates):
    user = user_rates[0]
    item_rates = dict(user_rates[1])

    #all items with their features
    item_features = list(filter(lambda x: item_rates.get([0]x, -1) != -1, grouped_features_array))
    features_ratings = list()
    for i in range(len(item_features)):
        item = item_features[i][0]
        temp = item_features[i][1]
        features_ratings = features_ratings + list(map(lambda x: (x, item_rates[item]), temp))

    #all features with their ratings
    features_ratings = sorted(features_ratings, key=lambda x: x[0])
    features_ratings = [(x,list(map(itemgetter(1),y))) for x,y in groupby(features_ratings, itemgetter(0))]
    result = list(map(lambda x: (x[0], mean_ratings(x[1])),features_ratings))

    return (user, result)

def intersects(dict, list):
    for f in list:
        if dict.get(f, -1) != -1:
            return True
    return False

def calculate_final_ratings(feats, prod):
    total = 0
    intersection = set(feats.keys()).intersection(prod)
    for f in prod:
        total = total + feats.get(f, 0)
    return total / (float(len(intersection)) + 0.5)

#for every test user calculates its model
#i = 0
f = open('submission2.csv', 'wt')

writer = csv.writer(f)
writer.writerow(('userId','RecommendedItemIds'))
#i = 0
for u in test_user_ratings.toLocalIterator():
    user_features_ratings = calculate_features_ratings(u)
    already_voted = test_user_ratings.filter(lambda y: u[0] == y[0]).flatMap(lambda x: x[1]).map(lambda x: x[0]).collect()
    #.filter(lambda y: u[0] == y[0]).flatMap(lambda x: x[1]).map(lambda x: x[0]).collect()
    dic_user_f_r = dict(user_features_ratings[1])
    #print(dic_user_f_r)
    #remove already voted, calculate products with common features, calculate ratings
    final_ratings = grouped_features.filter(lambda x: not x[0] in already_voted).filter(lambda x: intersects(dic_user_f_r, x[1])).map(lambda x: (x[0], calculate_final_ratings(dic_user_f_r, x[1])))
    #predictions = final_ratings.takeOrdered(5, lambda x: -x[1])
    predictions = final_ratings.sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).take(5)
    #.map(lambda x: x[0])
    #max_index = my_list.index(max_value)
    iterator = 0
    for i in range(5 - len(predictions)):
        while (item_ratings_mean[iterator] in already_voted) or (item_ratings_mean[iterator] in predictions):
            iterator = iterator + 1
        predictions = predictions + [item_ratings_mean[iterator]]
    writer.writerow((u[0], '{0} {1} {2} {3} {4}'.format(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])))
    #i+=1
    #print(i)

f.close()
