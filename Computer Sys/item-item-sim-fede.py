from itertools import combinations
import numpy as np
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split(",")
    return int(line[0]),int(line[1]),float(line[2])

def findItemPairs(user_id,items_with_rating):
    '''
    For each user, find all item-item pairs combos. (i.e. items with the same user)
    '''
    for item1,item2 in combinations(items_with_rating,2):
        yield (item1[0],item2[0]),(item1[1],item2[1])

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


def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0


train_rdd = sc.textFile("data/train.csv")
train_header = train_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(parseVector)

users_ratings = train_clean_data.map(lambda x: (x[0], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
users_ratings_mean = dict(users_ratings.mapValues(lambda x: (x[0] / x[1])).collect())
'''
Obtain the sparse user-item matrix
    user_id -> [(item_id_1, rating_1),
               [(item_id_2, rating_2),
                ...]
'''
user_item_pairs = train_clean_data.map(lambda x: (x[0],(x[1], x[2] - users_ratings_mean[x[0]]))).groupByKey().cache()

'''
Get all item-item pair combos
    (item1,item2) ->    [(item1_rating,item2_rating),
                         (item1_rating,item2_rating),
                         ...]
'''

pairwise_items = user_item_pairs.filter(
    lambda p: len(p[1]) > 1).map(
    lambda p: findItemPairs(p[0],p[1])).groupByKey()

'''
Calculate the cosine similarity for each item pair
    (item1,item2) ->    (similarity,co_raters_count)
'''

item_sims = pairwise_items.map(
    lambda p: calcSim(p[0],p[1])).collect()
item_sims[:50]
