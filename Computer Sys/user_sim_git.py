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

def calcSimEuclid(user_pair,rating_pairs,m):
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

    euc_sim = np.sqrt(squared_errors_sum) +(m-n)/n
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
            return int(users_nr_ratings[i][1])
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




'''
Calculate the cosine similarity for each user pair:
    (user1,user2) ->    (similarity,co_raters_count)
'''
users_ratings_count = item_user.map(lambda x: (x[1][0],1)).reduceByKey(lambda x,y: x+y).collect()

'''user_pair_sims = user_item_rating_pairs.map(
    lambda p: calcSim(p[0],p[1],getUserNrRatings(p[0][0],users_ratings_count)))'''

#taken users to raccomend
test_rdd= sc.textFile("data/test.csv")
test_header= test_rdd.first()
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))
useful_user_array=test_clean_data.map( lambda x: int(x[0])).collect()

user_pairs_euclidean=user_item_rating_pairs.filter(lambda x: x[0][0] in useful_user_array).map(
    lambda p: calcSimEuclid(p[0],p[1],getUserNrRatings(p[0][0],users_ratings_count)))

user_pairs_euclidean.take(18)

user_pairs_euclidean.saveAsTextFile('users_similarities2.csv')







'''everything already computed, go on from here but remember to run the imports'''














import csv
from scipy import linalg, sparse
import numpy as np
from itertools import groupby
from operator import itemgetter

sc = SparkContext.getOrCreate()

#ciaopalo
def intersects(dict, list):
    for f in list:
        if dict.get(f, -1) != -1:
            return True
    return False

#self explainatory
def calculate_final_ratings(feats, prod):
    total = 0
    intersection = set(feats.keys()).intersection(prod)
    for f in prod:
        total = total + feats.get(f, 0)
    return total / (float(len(intersection)) + 0.5)

#ret mean of a list
def mean_ratings(rates):
    return sum(rates) / float(len(rates))


#returns all the features voted by the user
def calculate_features_ratings(user_rates):
    user = user_rates[0]
    item_rates = dict(user_rates[1])

    item_features = list(filter(lambda x: item_rates.get(x[0], -1) != -1, grouped_features_array))
    features_ratings = list()
    for i in range(len(item_features)):
        item = item_features[i][0]
        temp = item_features[i][1]
        features_ratings = features_ratings + list(map(lambda x: (x, item_rates[item]), temp))
    #[item for item in temp if item[0] == 1][0]

    features_ratings = sorted(features_ratings, key=lambda x: x[0])
    features_ratings = [(x,list(map(itemgetter(1),y))) for x,y in groupby(features_ratings, itemgetter(0))]
    result = list(map(lambda x: (x[0], mean_ratings(x[1])),features_ratings))


    return (user, result)

#find the K-nearest neighbor of the selected user
def findKNN(k,similarities,user):

    user_sim = similarities.filter(lambda x: x[0]==user).sortBy(lambda x: x[2], ascending=True).map(lambda x: x[1]).collect()

    return user_sim



#parsing file di similarities salvato
def parse_KNN(line):
    line_no_simbols = line.replace("(", "").replace(")", "").replace(" ", "")
    elements = line_no_simbols.split(",")
    return ((int(elements[0]),int(elements[1]),float(elements[2])))

all_similarities = sc.textFile("users_similarities2.csv")
similarities_clean = all_similarities.map(parse_KNN)


#taken users to raccomend
test_rdd= sc.textFile("data/test.csv")
test_header= test_rdd.first()
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))
test_users=test_clean_data.map( lambda x: int(x[0])).collect()

#taken the whole training set
train_rdd = sc.textFile("data/train.csv")
train_header = train_rdd.first()
train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), int(x[2])))

#taken the item features informations
icm_rdd = sc.textFile("data/icm.csv")
icm_header = icm_rdd.first()
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))


#taken the features of every item
grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_features.cache()

# taken the items of every feature
grouped_items = icm_clean_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_items.cache()

#taken user (item, rating)
grouped_rates = train_clean_data.map(lambda x: (x[0],(x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_rates.cache()

#for every item all ratings shrinked
item_ratings = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
shrinkage_factor = 20
item_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()

#taken all ratings of the users to raccomend
def is_in_test(user):
    return user[0] in test_users
test_user_ratings = grouped_rates.filter(is_in_test)
test_user_ratings.cache()

#?????
grouped_features_array = grouped_features.collect()


k=20
pupo=0




#stampiamo i KNN
f = open('losKNN20.csv', 'wt')
writer = csv.writer(f)


for user in test_user_ratings.sortByKey().toLocalIterator():

    #accordingly to the KNN users find the items to which predict the rate
    KNN = findKNN(k,similarities_clean,user[0])
    '''items_of_similar_users = train_clean_data.filter(lambda x:x[0] in KNN).map(lambda x: x[1]).collect()
    items_of_similar_users= list(set(items_of_similar_users))

    dic_user_f_r = dict(user_features_ratings[1])


    #gets the evaluation of the ratings of every feature
    user_features_ratings = calculate_features_ratings(user)

    #gets what the current user already voted
    already_voted = test_user_ratings.filter(lambda y: user[0] == y[0]).flatMap(lambda x: x[1]).map(lambda x: x[0]).collect()


    #ccomputes ratings of items voted by similar users
    final_ratings = grouped_features.filter(lambda x: (not x[0] in already_voted) and (x[0] in items_of_similar_users) ).filter(lambda x: intersects(dic_user_f_r, x[1])).map(lambda x: (x[0], calculate_final_ratings(dic_user_f_r, x[1])))

    #sort the rated items in  order of rating and takes the best ones
    predictions = final_ratings.sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).take(5)

    #if the prediction cannot find enaugh items, the non voted top populars are provided
    iterator = 0
    for i in range(5 - len(predictions)):
        while item_ratings_mean[iterator] in already_voted:
            iterator = iterator + 1
        predictions = predictions + [item_ratings_mean[iterator]]'''

    pupo +=1


    #stampiamo i knn
    writer.writerow((user[0],KNN))
    print(pupo)



f.close()


len(KNN)
len(items_of_similar_users)
items_of_similar_users
user_features_ratings
already_voted
final_ratings.take(10)
predictions
dic_user_f_r
