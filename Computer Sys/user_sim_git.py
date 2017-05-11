import sys
from itertools import combinations
import numpy as np
import pdb

from pyspark import SparkContext


def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split(",")
    return int(line[1]),(int(line[0]),float(line[2]))

def keyOnUserPair(item_id,user_and_rating_pair):
    '''
    Convert each item and co_rating user pairs to a new vector
    keyed on the user pair ids, with the co_ratings as their value.
    '''
    (user1_with_rating,user2_with_rating) = user_and_rating_pair
    user1_id,user2_id = user1_with_rating[0],user2_with_rating[0]
    user1_rating,user2_rating = user1_with_rating[1],user2_with_rating[1]
    return (user1_id,user2_id),(user1_rating,user2_rating)

def calcSim(user_pair,rating_pairs,m):
    '''
    For each user-user pair, return the specified similarity measure,
    along with co_raters_count.
    '''
    sum_xx, sum_xy, sum_yy, sum_x, sum_y, n = (0.0, 0.0, 0.0, 0.0, 0.0, 0)

    for rating_pair in rating_pairs:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_xx),np.sqrt(sum_yy)) + m/n
    return user_pair, cos_sim

def calcSimEuclid(user_pair,rating_pairs):
    '''
    For each user-user pair, return the specified similarity measure,
    along with co_raters_count.
    '''
    squared_errors_sum,  n = (0.0, 0)

    for rating_pair in rating_pairs:

        squared_errors_sum += np.power(np.float(rating_pair[0])-np.float(rating_pair[1]),2)

        # sum_y += rt[1]
        # sum_x += rt[0]
        n += 1

    euc_sim = np.sqrt(squared_errors_sum) +2/n
    return user_pair, euc_sim

def cosine(dot_product,rating_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors A, B
       dotProduct(A, B) / (norm(A) * norm(B))
    '''
    numerator = dot_product
    denominator = rating_norm_squared * rating2_norm_squared

    return (numerator / (float(denominator))) if denominator else 0.0

def getUserNrRatings(user,users_nr_ratings):

    for i in range(len(users_nr_ratings)):
        if users_nr_ratings[i][0] == user:
            return int(users_nr_ratings.pop(i)[1])
    return 0

sc = SparkContext.getOrCreate()
lines_not_filtered = sc.textFile("data/train.csv")
header = lines_not_filtered.first()
lines = lines_not_filtered.filter(lambda x: x != header)

'''
Parse the vector with item_id as the key:
    item_id -> (user_id,rating)
'''
item_user = lines.map(parseVector).cache()

users_ratings_count = item_user.map(lambda x: (x[1][0],1)).reduceByKey(lambda x,y: x+y).collect()


'''
Get co_rating users by joining on item_id:
    item_id -> ((user_1,rating),(user2,rating))
'''
item_user_pairs = item_user.join(item_user)



'''
Key each item_user_pair on the user_pair and get rid of non-unique
user pairs, then aggregate all co-rating pairs:
    (user1_id,user2_id) -> [(rating1,rating2),
                            (rating1,rating2),
                            (rating1,rating2),
                            ...]
'''
user_item_rating_pairs = item_user_pairs.map(
    lambda p: keyOnUserPair(p[0],p[1])).filter(
    lambda p: p[0][0] != p[0][1]).groupByKey()

u


'''
Calculate the cosine similarity for each user pair:
    (user1,user2) ->    (similarity,co_raters_count)
'''
users_ratings_count = item_user.map(lambda x: (x[1][0],1)).reduceByKey(lambda x,y: x+y).collect()
bob=getUserNrRatings(4,users_ratings_count)
bob



user_pair_sims = user_item_rating_pairs.map(
    lambda p: calcSim(p[0],p[1],getUserNrRatings(p[0][0],users_ratings_count)))

user_pairs_euclidean=user_item_rating_pairs.map(
    lambda p: calcSimEuclid(p[0],p[1],getUserNrRatings(p[0][0],users_ratings_count)))

user_pair_sims.take(18)
user_pairs_euclidean.take(1)
