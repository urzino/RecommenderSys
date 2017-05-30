from pyspark import SparkContext
import csv
from scipy import linalg, sparse
import numpy as np
from itertools import groupby
from functools import reduce
from operator import itemgetter
from collections import defaultdict

sc = SparkContext.getOrCreate()

def calculate_content_matrix(train_clean_data, icm_clean_data, test_clean_data):
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

    grouped_features.map(lambda x: (x[0], x[1], 1/ np.sqrt(len(x[1])))).map(lambda x: (x[0], [(item, x[2]) for item in x[1]]))

    tf_grouped_features = grouped_features.map(lambda x: (x[0], x[1], 1/ np.sqrt(len(x[1])))).map(lambda x: (x[0], [(item, x[2]) for item in x[1]]))
    tf_grouped_features_dic = sc.broadcast(dict(tf_grouped_features.collect()))
    tf_grouped_features_dic.value.get(11812)

    tf_item = tf_grouped_features.map(lambda x: (x[0], x[1][0][1])).collect()
    tf_item_dic = dict(tf_item)

    #for every features all its items
    #grouped_items = sc.parallelize([(1,[1,4]),(2,[1,2,4]),(3,[2,3]),(4,[2,3,4])])
    grouped_items = icm_clean_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
    grouped_items.take(10)
    grouped_items.cache()
    grouped_items_dic = dict(grouped_items.collect())

    idf_features = sc.broadcast(dict(grouped_items.map(lambda x: (x[0], np.log10(total_items / len(x[1])))).collect()))

    def group_items_tf(f_items):
        feature = f_items[0]
        items = f_items[1]
        return (feature, [(i, tf_item_dic.get(i, 0)) for i in items])
    tf_grouped_items = dict(grouped_items.map(group_items_tf).collect())

    #for every user all its ratings (item, rate)
    #grouped_rates = sc.parallelize([(1,[(1,8),(3,2)]),(2,[(1,2),(2,9),(3,7)]),(3,[(3,1),(4,10)])])
    grouped_rates = train_clean_data.map(lambda x: (x[0],(x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
    grouped_rates.cache()
    grouped_rates_dic = dict(train_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))).collect())
    #for every item all its ratings
    item_ratings = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
    #item_ratings.take(10)
    shrinkage_factor = 5
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
            result += list(map(lambda x: (x[0], rating * x[1]), tf_grouped_features_dic.value.get(item, [])))
        result.sort(key = lambda x: x[0])
        result = [(x,sum([z[1] for z in y])) for x,y in groupby(result, itemgetter(0))]
        return (user_id, result)

    def calculate_user_idf(user_tf):
        user_id = user_tf[0]
        ratings = user_tf[1]
        result = list()
        for feature, rating in ratings:
            result += [(feature, rating * idf_features.value.get(feature, 0))]
        return(user_id, result)

    def calculate_final_percentages(user_tf, n):
        user_id = user_tf[0]
        ratings = user_tf[1]
        items_dict = defaultdict(int)
        already_voted = grouped_rates_dic[user_id]
        for feature, rating in ratings:
            items_with_f = tf_grouped_items.get(feature, [])
            for item, tf in items_with_f:
                items_dict[item] += tf * rating

        scored_items = [(total,item) for item,total in items_dict.items() if total != 0 and not item in already_voted]

        # sort the scored items in ascending order
        scored_items.sort(reverse=True)

        # take out the item score
        #ranked_items = [x[1] for x in scored_items]
        ranked_items = scored_items
        if n == -1:
            return user_id,ranked_items
        return user_id,ranked_items[:n]

    users_tf = test_user_ratings.filter(lambda x: x[0] in test_users).map(calculate_ratings)

    return users_tf.map(calculate_user_idf).map(lambda x: calculate_final_percentages(x, -1))

def print_output():
    #.collect()


    f = open('submission2.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(('userId','RecommendedItemIds'))
    #i = 0
    for u in users_final_ratings:
        already_voted = grouped_rates_dic[u[0]]
        predictions = u[1]
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
