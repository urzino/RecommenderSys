
# coding: utf-8

# In[35]:

from pyspark import SparkContext
import csv
from scipy import linalg, sparse
import numpy as np
from itertools import groupby
from itertools import combinations, permutations
from functools import reduce
from operator import itemgetter
from collections import defaultdict

sc = SparkContext.getOrCreate()


# In[36]:

def findItemPairs(feature_id,items):
    '''
    For each user, find all item-item pairs combos. (i.e. items with the same user)
    '''
    result = list()
    for item1,item2 in permutations(items,2):
        result += [((item1[0],item2[0]),(item1[1],item2[1]))]
    return result

def calcSim(item_pair,rating_pairs):
    '''
    For each item-item pair, return the specified similarity measure,
    along with co_raters_count
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)

    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy))
    return item_pair, (cos_sim,n)

shrinkage_factor_cosine = 4

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared + shrinkage_factor_cosine
    return (numerator / (float(denominator))) if denominator else 0.0

def correlation(size, dot_product, rating_sum, \
            rating2sum, rating_norm_squared, rating2_norm_squared):
    '''
    The correlation between two vectors A, B is
      [n * dotProduct(A, B) - sum(A) * sum(B)] /
        sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }
    '''
    numerator = size * dot_product - rating_sum * rating2sum
    denominator = sqrt(size * rating_norm_squared - rating_sum * rating_sum) * \
                    sqrt(size * rating2_norm_squared - rating2sum * rating2sum)

    return (numerator / (float(denominator))) if denominator else 0.0

def keyOnFirstItem(item_pair,item_sim_data):
    '''
    For each item-item pair, make the first item's id the key
    '''
    (item1_id,item2_id) = item_pair
    return item1_id,(item2_id,item_sim_data)

def nearestNeighbors(item_id,items_and_sims,n):
    '''
    Sort the predictions list by similarity and select the top-N neighbors
    '''
    items_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return item_id, items_and_sims[:n]

def topNRecommendations(user_id,items_with_rating,item_sims,n):
    '''
    Calculate the top-N item recommendations for each user using the
    weighted sums method
    '''

    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item neighborhood
    totals = defaultdict(int)
    sim_sums = defaultdict(int)
    already_voted = grouped_rates_dic[user_id]

    '''
    items_ratings = dict(items_with_rating)
    for item in item_sims.keys():
        nearest_neighbors = item_sims.get(item,None)
        if nearest_neighbors:
            for (neighbor,(sim,count)) in nearest_neighbors:
                rating = items_ratings.get(item, 0)
                if rating != 0:
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim
                rating_neighbor = items_ratings.get(neighbor, 0)
                if rating_neighbor != 0:
                    totals[item] += sim * rating_neighbor
                    sim_sums[item] += sim
    '''
    for (item,rating) in items_with_rating:

        # lookup the nearest neighbors for this item
        nearest_neighbors = item_sims.get(item,None)
        if nearest_neighbors:
            for (neighbor,(sim,count)) in nearest_neighbors:
                if neighbor != item:

                    # update totals and sim_sums with the rating data
                    totals[neighbor] += sim * rating
                    sim_sums[neighbor] += sim

    # create the normalized list of scored items
    #/sim_sums[item]
    scored_items = [(total,item) for item,total in totals.items() if sim_sums[item] != 0 and not item in already_voted]

    # sort the scored items in ascending order
    scored_items.sort(reverse=True)

    # take out the item score
    #ranked_items = [x[1] for x in scored_items]
    ranked_items = scored_items
    if n == -1:
        return user_id,ranked_items
    return user_id,ranked_items[:n]


train_rdd = sc.textFile("../data/train.csv")
icm_rdd = sc.textFile("../data/icm_fede.csv")
test_rdd= sc.textFile("../data/target_users.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))



# In[37]:


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

tf_grouped_features = grouped_features.map(lambda x: (x[0], x[1], 1/ np.sqrt(len(x[1])))).map(lambda x: (x[0], [(item, x[2]) for item in x[1]]))
tf_grouped_features_dic = sc.broadcast(dict(tf_grouped_features.collect()))

tf_item = tf_grouped_features.map(lambda x: (x[0], x[1][0][1])).collect()
tf_item_dic = dict(tf_item)

#for every features all its items
#grouped_items = sc.parallelize([(1,[1,4]),(2,[1,2,4]),(3,[2,3]),(4,[2,3,4])])
grouped_items = icm_clean_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_items.cache()
grouped_items_dic = dict(grouped_items.collect())
idf_features = sc.broadcast(dict(grouped_items.map(lambda x: (x[0], np.log10(total_items / len(x[1])))).collect()))
idf_features.value.get(1)
def tf_idf(item_features):
    item_id = item_features[0]
    result = list()
    for feature, tf in item_features[1]:
        result += [(feature, tf * idf_features.value.get(feature))]
    return item_id, result

tf_idf_items = tf_grouped_features.map(tf_idf).cache()

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

grouped_items_f = tf_idf_items.flatMap(lambda x: [(f, (x[0], tf_idf_f)) for f, tf_idf_f in x[1]]).groupByKey().map(lambda x: (x[0], list(x[1]))).cache()

pairwise_items = grouped_items_f.flatMap(
    lambda p: findItemPairs(p[0],p[1])).groupByKey()
#Calculate the cosine similarity for each item pair and select the top-N nearest neighbors:(item1,item2) ->    (similarity,co_raters_count)

item_sims = pairwise_items.map(
    lambda p: calcSim(p[0],list(p[1]))).map(lambda p: keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: nearestNeighbors(p[0],list(p[1]),50000)).collect()
#Preprocess the item similarity matrix into a dictionary and store it as a broadcast variable:
#item_sims
item_sim_dict = {}
for (item,data) in item_sims:
    item_sim_dict[item] = data

isb = sc.broadcast(item_sim_dict)
#Calculate the top-N item recommendations for each user user_id -> [item1,item2,item3,...]

#user_item_recs = user_item_pairs.filter(lambda x: x[0] in test_users).map(lambda p: topNRecommendations(p[0],p[1],isb.value,5)).sortByKey().collect()
user_item_recs = train_clean_data.filter(lambda x: x[0] in test_users).map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda p: (p[0],list(p[1]))).map(lambda p: topNRecommendations(p[0],p[1],isb.value,5)).sortByKey().collect()
f = open('../submission2.csv', 'wt')

writer = csv.writer(f)
writer.writerow(('userId','RecommendedItemIds'))

for u in user_item_recs:
    predictions = u[1]
    iterator = 0
    already_voted = grouped_rates_dic[u[0]]
    for i in range(5 - len(predictions)):
        while (item_ratings_mean[iterator] in already_voted) or (item_ratings_mean[iterator] in predictions):
            iterator = iterator + 1
        predictions = predictions + [item_ratings_mean[iterator]]
    writer.writerow((u[0], '{0} {1} {2} {3} {4}'.format(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])))
    #i+=1
    #print(i)

f.close()
