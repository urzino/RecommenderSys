from pyspark import SparkContext
import csv
#from pyspark.mllib.linalg import Matrix, Matrices
from scipy import linalg, sparse
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
    item_features = list(filter(lambda x: item_rates.get(x[0], -1) != -1, grouped_features_array))
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

'''
temp2 = grouped_rates.take(1)[0][1]
temp3 = list(map(lambda x: x[0], temp2))

temp3
temp = dict(temp2)
#y = grouped_features.filter(lambda x: temp.get(x[0], -1) != -1).flatMap(lambda x: [(f, temp[x[0]]) for f in x[1]]).groupByKey().map(lambda x: (x[0], mean_ratings(x[1])))
y = list(filter(lambda x: temp.get(x[0], -1) != -1, grouped_features_array))
y
y2 = list()
for i in range(len(y)):
    item = y[i][0]
    temp4 = y[i][1]
    y2 = y2 + list(map(lambda x: (x, temp[item]), temp4))

temp
#[item for item in temp if item[0] == 1][0]

y2 = sorted(y2, key=lambda x: x[0])
y3 = [(x,list(map(itemgetter(1),y))) for x,y in groupby(y2, itemgetter(0))]
y4 = list(map(lambda x: (x[0], mean_ratings(x[1])),y3))
y2
y4
y3
'''

'''
for x in test_user_ratings.take(1):
    temp = calculate_features_ratings(x)
    already_voted = test_user_ratings.filter(lambda y: x[0] == y[0]).flatMap(lambda x: x[1]).map(lambda x: x[0]).collect()
    print(temp)
    print(already_voted)
    temp3 = dict(temp[1])
    #remove already voted, calculate products with common features, calculate ratings
    temp2 = grouped_features.filter(lambda x: not x[0] in already_voted).filter(lambda x: intersects(temp3, x[1])).map(lambda x: (x[0], calculate_final_ratings(temp3, x[1])))
    print(temp2.count())
    #filter(lambda x: (temp3.get(temp3[f[0]], -1) != -1 for f in x[1]))
    #filter(lambda x: intersects(temp3, x[1]))
    print(temp2.takeOrdered(20, lambda x: -x[1]))
'''

'''
#15374 utente max
#user_array = train_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False)
#user_array.take(10)

#37141 item max
#item_array = icm_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False)
#item_array.take(10)

#19715 feature max
#features_array = icm_clean_data.map( lambda x: int(x[1])).sortBy(lambda x: x, ascending=False)
#features_array.take(10)
#tutte le righe degli utenti da riempire
user_array = train_clean_data.map(lambda x: int(x[0]))
user_array.take(10)
total_users = user_array.count()
print(total_users)

#tutte le colonne degli item da riempire
item_array = train_clean_data.map(lambda x: int(x[1]))
item_array.take(10)
print(item_array.count())

#tutti i valori dei rating ordinati come gli array sopra
rating_array = train_clean_data.map(lambda x: float(x[2]))
rating_array.take(10)
print(rating_array.count())

#tutti gli item anche se non sono stati votati
items = icm_clean_data.map(lambda x: int(x[0]))
total_items = items.distinct().count()
print(total_items)
'''

'''
#U_dimension=user_array.take(1)[0]+2
I_dimension=item_array.take(1)[0]+2
F_dimension=features_array.take(1)[0]+2

IxF=np.zeros(shape=(I_dimension,F_dimension))
for item in icm_clean_data.toLocalIterator():
    IxF[item[0]][item[1]] = 1

f = open('submission2.csv', 'wt')
try:
    writer = csv.writer(f)
    writer.writerow(('userId','RecommendedItemIds'))
    k = 0
    for user in useful_user_array.toLocalIterator():

        UxI_weighted=np.zeros(shape=(1,I_dimension))
        UxI_one=np.zeros(shape=(1,I_dimension))

        for rate in train_clean_data.toLocalIterator():
            if rate[0]==user:
                UxI_weighted[0][rate[1]]=rate[2]
                UxI_one[0][rate[1]]=1

        UxF=np.dot(UxI_weighted,IxF)
        UxF_counter=np.dot(UxI_one,IxF)
        for col in range(F_dimension):
            if UxF_counter[0][col]!=0:
                UxF[0][col]=UxF[0][col]/UxF_counter[0][col]

        UxI_pred = np.dot(UxF, np.transpose(IxF))
        for item in grouped_features.toLocalIterator():
            UxI_pred[0][item[0]] = UxI_pred[0][item[0]] / len(item[1])

        for i in range(I_dimension):
            if UxI_one[0][i] == 1:
                UxI_pred[0][i] == 0

        predictions = np.zeros(shape=(1,5))
        for i in range(5):
            top_item = np.argmax(UxI_pred, axis = 1)[0]
            predictions[0][i] = top_item
            UxI_pred[0][top_item] = 0
        #break
        writer.writerow((user, '{0} {1} {2} {3} {4}'.format(predictions[0][0], predictions[0][1], predictions[0][2], predictions[0][3], predictions[0][4])))
        k = k + 1
        print(k)
finally:
    f.close()
#UxI_pred
#predictions
#valore massimo per riga
#np.argmax(UxI_pred, axis = 1)[0]
'''
