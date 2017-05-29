from pyspark import SparkContext
from operator import add
from itertools import groupby
from operator import itemgetter
sc = SparkContext.getOrCreate()

rdd1 = sc.parallelize([(4, [(2,3),(5,4),(6,8),(9,7)]),(5, [(2,3),(5,4),(6,8),(9,7)]),(6, [(2,3),(5,4),(6,8),(9,7)]),(7, [(2,3),(5,4),(6,8),(9,7)])])
rdd2 = sc.parallelize([(4, [(2,3),(5,4),(6,8),(9,7)]),(5, [(2,3),(5,4),(6,8),(9,7)]),(6, [(2,3),(5,4),(6,8),(9,7)]),(7, [(2,3),(5,4),(6,8),(9,7)])])

def add_weights(user_rates):
    return user_rates[0],[(item, rating * 0.5)for item,rating in user_rates[1]]

rdd2a = rdd2.map(add_weights)

def union_ratings(user_rates):
    ratings = list()
    for rates in user_rates[1]:
        ratings += rates
    return user_rates[0], ratings

rdd3 = rdd1.union(rdd2a).reduceByKey(add)

def sum_weights(user_rates):
    ratings = user_rates[1]
    ratings = sorted(ratings, key=lambda x: x[0])
    ratings = [(x,list(map(itemgetter(1),y))) for x,y in groupby(ratings, itemgetter(0))]
    result = list(map(lambda x: (x[0], sum(x[1])),ratings))
    return user_rates[0], result

rdd4 = rdd3.map(sum_weights)
rdd4.take(10)
