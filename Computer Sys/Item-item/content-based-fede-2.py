from pyspark import SparkContext
import csv
from scipy import linalg, sparse
import numpy as np
from itertools import groupby
from functools import reduce
from operator import itemgetter

sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("../data/train.csv")
icm_rdd = sc.textFile("../data/icm_fede.csv")
test_rdd= sc.textFile("../data/target_users.csv")

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
total_items = grouped_features.count()
grouped_features_arr = grouped_features.collect()
grouped_features_dic = sc.broadcast(dict(grouped_features.collect()))

tf_grouped_features = sc.broadcast(dict(grouped_features.map(lambda x: (x[0], x[1], 1/ np.sqrt(len(x[1])))).map(lambda x: (x[0], [(item, x[2]) for item in x[1]])).collect()))

#for every features all its items
#grouped_items = sc.parallelize([(1,[1,4]),(2,[1,2,4]),(3,[2,3]),(4,[2,3,4])])
grouped_items = icm_clean_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
#grouped_items.take(10)
grouped_items.cache()
grouped_items_dic = dict(grouped_items.collect())

idf_features = sc.broadcast(dict(grouped_items.map(lambda x: (x[0], np.log10(total_items / len(x[1])))).collect()))
#for every user all its ratings (item, rate)
#grouped_rates = sc.parallelize([(1,[(1,8),(3,2)]),(2,[(1,2),(2,9),(3,7)]),(3,[(3,1),(4,10)])])
grouped_rates = train_clean_data.map(lambda x: (x[0],(x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_rates.cache()

#for every item all its ratings
item_ratings = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
#item_ratings.take(10)
shrinkage_factor = 10
item_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()
#.map(lambda x: x[0])
#return only test users
def is_in_test(user):
    return user[0] in test_users

test_user_ratings = grouped_rates.filter(is_in_test).sortByKey()
test_user_ratings.cache()

test_voted_items = test_user_ratings.map(lambda x: (x[0], [item for item, rate in x[1]])).collect()
test_voted_items_dic = dict(test_voted_items)

test_user_features = grouped_rates.map(lambda x: (x[0], [grouped_features_dic.value.get(item, []) for item, rating in x[1]])).map(lambda x: (x[0], set(reduce(lambda x,y: x+y, x[1]))))
test_user_features_dic = sc.broadcast(dict(test_user_features.collect()))

#test_user_features_dic.value
def calculate_ratings(user_rates):
    user_id = user_rates[0]
    i_rates = user_rates[1]
    result = list()
    for item, rating in i_rates:
        result += list(map(lambda x: (x[0], rating * x[1]), tf_grouped_features.value.get(item, [])))
    result.sort(key = lambda x: x[0])
    result = [(x,sum([z[1] for z in y])) for x,y in groupby(result, itemgetter(0))]
    return (user_id, result)

def calculate_final_percentages(user_tf):
    user_id = user_tf[0]
    result = list()
    fs_dic = dict(user_tf[1])
    already_voted = test_voted_items_dic.get(user_id, [])
    items = filter(lambda x: x[0] not in already_voted and len(set(x[1]).intersection(test_user_features_dic.value[user_id])) > 0, grouped_features_arr)
    for item, fs in items:
        tf_dic = dict(tf_grouped_features.value[item])
        result += [(item, sum([fs_dic.get(f, 0) * idf_features.value.get(f,0) * tf_dic.get(f, 0) for f in fs]))]
    result.sort(key = lambda x: -x[1])
    result = list(map(lambda x: x[0], result))
    return (user_id, result[:5])

users_tf = test_user_ratings.map(calculate_ratings)
users_tf.take(5)
users_final_ratings = users_tf.map(calculate_final_percentages).collect()
users_final_ratings[:5]
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
