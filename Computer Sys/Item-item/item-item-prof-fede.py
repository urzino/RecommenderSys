from collections import defaultdict
from itertools import combinations, permutations
import numpy as np
import random
import csv
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

def parseVector(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Converts each rating to a float
    '''
    line = line.split(",")
    return int(line[0]),int(line[1]),float(line[2])

def parseFeatures(line):
    '''
    Parse each line of the specified data file and gets all the features ratings for each user
    '''
    ratings = list()
    user_line, ratings_line = line.split("[")
    user_line = user_line.replace("(", "").replace(",", "").replace(" ", "")
    ratings_line = ratings_line.replace("(", "").replace("]","").replace(" ", "")
    ratings_line = ratings_line.split("),")
    for rate in ratings_line:
        rate_tuple = rate.replace(")", " ").split(",")
        ratings += [(int(rate_tuple[0]), float(rate_tuple[1]))]
    user_id = int(user_line)
    return (user_id, ratings)

def parse_icm(line):
    item, feature = line.split(",")
    return int(item), int(feature)

def parseSimilarities(line):
    line = line.replace("(","").replace(")","").replace(" ","").split(",")
    return ((int(line[0]),int(line[1])),(float(line[2]), int(line[3])))

def sampleInteractions(user_id,items_with_rating,n):
    '''
    For users with # interactions > n, replace their interaction history
    with a sample of n items_with_rating
    '''
    if len(items_with_rating) > n:
        return user_id, random.sample(items_with_rating,n)
    else:
        return user_id, items_with_rating

def findItemPairs(user_id,items_with_rating):
    '''
    For each user, find all item-item pairs combos. (i.e. items with the same user)
    '''
    result = list()
    for item1,item2 in permutations(items_with_rating,2):
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
    ranked_items = [x[1] for x in scored_items]
    #ranked_items = scored_items
    return user_id,ranked_items[:n]

def find_already_voted(user_predictions, user_rates):
    for pred in user_predictions:
        if pred[1] in user_rates:
            return False
    return True

train_rdd = sc.textFile("../data/train.csv")
test_rdd= sc.textFile("../data/target_users.csv")
features_ratings_rdd = sc.textFile("features_rates.csv")
icm_rdd = sc.textFile("../data/icm_fede.csv")

train_header = train_rdd.first()
test_header= test_rdd.first()
icm_header = icm_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(parseVector)
test_clean_data = test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))
features_clean_data = features_ratings_rdd.map(parseFeatures)
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(parse_icm)
#train_clean_data = sc.parallelize([(1,1,2.5),(1,2,-1.5),(1,4,-0.5),(1,5,-0.5),(2,1,-2.6),(2,2,1.4),(2,3,-1.6),(2,4,1.4),(2,5,1.4),(3,1,-1.5),(3,3,-0.5),(3,4,1.5),(3,5,0.5),(4,1,0.25),(4,2,-0.75),(4,3,1.25),(4,4,-0.75)])

train_clean_data.cache()
test_clean_data.cache()
features_clean_data.cache()
icm_clean_data.cache()

test_users=test_clean_data.map( lambda x: int(x[0])).collect()
#test_users = [1,2,3,4]

grouped_rates = train_clean_data.filter(lambda x: x[0] in test_users).map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
grouped_rates_dic = dict(grouped_rates)

item_features = icm_clean_data.groupByKey().map(lambda x: (x[0], list(x[1]))).cache()
item_features_dic = sc.broadcast(dict(item_features))
#for every item all its ratings
item_ratings = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
shrinkage_factor = 20
item_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()

#Obtain the sparse user-item matrix: user_id -> [(item_id_1, rating_1),  [(item_id_2, rating_2),

#user_item_pairs = train_clean_data.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda p: (p[0],list(p[1]))).cache()

#Get all item-item pair combos: (item1,item2) ->    [(item1_rating,item2_rating),(item1_rating,item2_rating),

pairwise_items = item_features.flatMap(
    lambda p: findItemPairs(p[0],p[1])).groupByKey()

#Calculate the cosine similarity for each item pair and select the top-N nearest neighbors:(item1,item2) ->    (similarity,co_raters_count)

item_sims = pairwise_items.map(
    lambda p: calcSim(p[0],p[1])).map(lambda p: keyOnFirstItem(p[0],p[1])).groupByKey().map(lambda p: nearestNeighbors(p[0],list(p[1]),50)).collect()
#Preprocess the item similarity matrix into a dictionary and store it as a broadcast variable:
#item_sims
item_sim_dict = {}
for (item,data) in item_sims:
    item_sim_dict[item] = data

isb = sc.broadcast(item_sim_dict)
#Calculate the top-N item recommendations for each user user_id -> [item1,item2,item3,...]

#user_item_recs = user_item_pairs.filter(lambda x: x[0] in test_users).map(lambda p: topNRecommendations(p[0],p[1],isb.value,5)).sortByKey().collect()
user_item_recs = train_clean_data.filter(lambda x: x[0] in test_users).map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda p: (p[0],list(p[1]))).map(lambda p: topNRecommendations(p[0],p[1],isb.value,5)).sortByKey().collect()

def parseSubmission(line):
    user, items = line.split(",")
    items = items.split(" ")
    return (int(user), [int(item) for item in items])

submission_rdd = sc.textFile("submission.csv")
submission_header = submission_rdd.first()
submission_clean_data = submission_rdd.filter(lambda x: x != submission_header).map(parseSubmission).collect()
#submission_clean_data[:2]
submission_clean_data_dic = dict(submission_clean_data)

f = open('../submission2.csv', 'wt')

writer = csv.writer(f)
writer.writerow(('userId','RecommendedItemIds'))

for u in user_item_recs:
    predictions = u[1]
    iterator = 0
    already_voted = grouped_rates_dic[u[0]]
    content_based_items = submission_clean_data_dic[u]
    for i in range(5 - len(predictions)):
        while (content_based_items[iterator] in already_voted) or (content_based_items[iterator] in predictions):
            iterator = iterator + 1
        predictions = predictions + [content_based_items[iterator]]
    writer.writerow((u[0], '{0} {1} {2} {3} {4}'.format(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])))
    #i+=1
    #print(i)

f.close()
