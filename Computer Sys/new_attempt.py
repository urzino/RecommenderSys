from pyspark import SparkContext
import csv
from pyspark.mllib.linalg import Matrix, Matrices
from scipy import linalg
import numpy as np
import math
import random



def calculateDist(user1,user2):
    squared_distance_sum=0


    #for rating1 in user1.toLocalIterator():
        #for rating2 in user2.toLocalIterator():
            #if rating1[1]==rating2[1]:
            #    squared_distance_sum=squared_distance_sum + math.pow((rating1[2]-rating2[2]),2)
            #    print (squared_distance_sum)

    squared_distance_sum=random.uniform(0,90)

    return  math.sqrt(squared_distance_sum)

def findKNN(K,user1,users_ratings,users_all):
    KNN = [0] * K
    KNN_closeness = [None] * K

    user1_ratings = users_ratings.filter(lambda x: x[0]==user1)

    #se user1 non ha fatto ratings ritorna none

    if user1_ratings.isEmpty():
        return None
    for user2 in users_all.toLocalIterator():
        user2_ratings= users_ratings.filter( lambda x: x[0] == user2)
        if user2_ratings.isEmpty():
            continue
        #se user2 non ha fatto rating non farci nulla

        distance = calculateDist(user1_ratings,user2_ratings)


        if 0 in KNN:
            for j in range(K):
                if KNN[j]==0:
                    KNN[j]=user2
                    KNN_closeness[j]=distance
                    break
        else:
            actual_max_dist= max(KNN_closeness)
            for i in range(K):
                if KNN_closeness[i]==actual_max_dist:
                    KNN[i]=user2
                    KNN_closeness[i]=distance
                    break



    return KNN


sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("data/train.csv")
icm_rdd = sc.textFile("data/icm.csv")
test_rdd= sc.textFile("data/test.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), int(x[2])))
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))

user_array = train_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False)
item_array= icm_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False)
features_array = icm_clean_data.map( lambda x: int(x[1])).sortBy(lambda x: x, ascending=False)
useful_user_array=test_clean_data.map( lambda x: int(x[0])) #users to make recs

grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))) #item, all features

#Set k of the KNN of every useful user you want to find

k=20



#U_dimension=user_array.take(1)[0]+2
I_dimension=item_array.take(1)[0]+2
F_dimension=features_array.take(1)[0]+2

useful_user_array.take(10)

for user2 in user_array.take(1):
    user2_ratings= train_clean_data.filter( lambda x: x[0] == user2)
    print(user2_ratings.take(100))

for user in useful_user_array.toLocalIterator():


    KNN = findKNN(k,user,train_clean_data,user_array)
    break
KNN
