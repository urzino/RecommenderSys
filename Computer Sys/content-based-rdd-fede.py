from pyspark import SparkContext
import csv
import numpy as np
from itertools import groupby
from operator import itemgetter

sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("data/train.csv")
icm_rdd = sc.textFile("data/icm.csv")
test_rdd= sc.textFile("data/test.csv")

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

#for every user the mean of its ratings
users_ratings = train_clean_data.map(lambda x: (x[0], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
users_ratings_mean = dict(users_ratings.mapValues(lambda x: (x[0] / x[1])).collect())

#for every item all its ratings
items_ratings = train_clean_data.map(lambda x: (x[1], (x[0], x[2] - users_ratings_mean[x[0]]))).groupByKey().map(lambda x: (x[0], list(x[1])))
items_ratings.cache()
#items_ratings.take(10)
#items_ratings.filter(lambda x: len(x[1]) == 1).map(lambda x: (x[1],x[0])).sortBy(lambda x: x[0]).take(100)
items_ratings_arr = items_ratings.collect()

#returns the intersection of dicts of (user, rating)
def calculate_intersection(dict1, dict2):
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    #print(keys1, keys2)
    return list(set(keys1).intersection(set(keys2)))

#returns an item and a list of similar items with weights
def calculate_similarities(item1):
    similarities = list()
    for item2 in items_ratings_arr:
        if item2[0] != item1[0]:
            dict1 = dict(item1[1])
            dict2 = dict(item2[1])
            intersections = calculate_intersection(dict1, dict2)
            if len(intersections) == 0:
                continue
            #if len(intersections) != 0:
            #print(intersections)
            numerator = 0
            denominator1 = 0
            denominator2 = 0
            for user in intersections:
                rating1 = float(dict1[user])
                rating2 = float(dict2[user])
                numerator += rating1 * rating2
                denominator1 += np.power(rating1,2)
                denominator2 += np.power(rating2, 2)
            if denominator1 == 0 or denominator2 == 0:
                print("ciao")
                continue
            denominator1 = np.sqrt(denominator1)
            denominator2 = np.sqrt(denominator2)
            result = numerator / (denominator1 * denominator2)
            similarities += [(item2[0], result)]
            #numerator = sum([float(dict1[user])*float(dict2[user]) for user in intersections])
            #denominator1 = np.sqrt(sum([np.power(float(dict1[user]), 2) for user in intersections]))
            #denominator2 = np.sqrt(sum([np.power(float(dict2[user]), 2) for user in intersections]))
    return (item1[0], similarities) #similarities

items_similarities = items_ratings.map(calculate_similarities)
items_similarities.take(20)

for item1 in items_ratings_arr[:10]:
    i = calculate_similarities(item1)
i
item1 = items_ratings_arr[1]
item1
dic1 = dict(item1[1])
dic1
sum([dic1[key] for key in list(dic1.keys())])
