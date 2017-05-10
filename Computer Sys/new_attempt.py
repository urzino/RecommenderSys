from pyspark import SparkContext
import csv
from pyspark.mllib.linalg import Matrix, Matrices
from scipy import linalg
import numpy as np
import math
import random

def getUserRatings(user,all_users):
    us = []
    for i in range(len(all_users)-1):
         if all_users[i][0] == user:
             us.append(all_users[i])


    return us



def calculateDist(us1,us2):
    squared_distance_sum=0


    '''for rating1 in user1.toLocalIterator():

        rating2 = user2.filter(lambda x: x[1]==rating1[1])
        if rate.count() != 1:                continue


        squared_distance_sum=squared_distance_sum + math.pow((rating1[2]-rating2.take(1)[0][2]),2)
        print (squared_distance_sum)'''




    for i in range(len(us1)):
        for j in range(len(us2)):
            if us1[i][1]==us2[j][1]:

                squared_distance_sum=squared_distance_sum + math.pow((us1[i][2]-us2[j][2]),2)


    #squared_distance_sum=random.uniform(0,90)

    return  math.sqrt(squared_distance_sum)

def findKNN(K,user1,users_ratings,users_all):
    KNN = [0] * K
    KNN_closeness = [None] * K

    user1_ratings = getUserRatings(user1,users_ratings)

    #se user1 non ha fatto ratings ritorna none

    if len(user1_ratings) == 0:
        return None

    for k in range(len(users_all)):
        user2_ratings = getUserRatings(users_all[k],users_ratings)

        if len(user2_ratings) == 0:
            continue
        #se user2 non ha fatto rating non farci nulla

        distance = calculateDist(user1_ratings,user2_ratings.copy())


        if 0 in KNN:
            for j in range(K):
                if KNN[j]==0:
                    KNN[j]=users_all[k]
                    KNN_closeness[j]=distance
                    break
        else:
            actual_max_dist= max(KNN_closeness)
            for i in range(K):
                if KNN_closeness[i]==actual_max_dist:
                    KNN[i]=users_all[k]
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

user_array = train_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False).distinct()
item_array= icm_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False)
features_array = icm_clean_data.map( lambda x: int(x[1])).sortBy(lambda x: x, ascending=False)
useful_user_array=test_clean_data.map( lambda x: int(x[0])) #users to make recs

grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))) #item, all features

#Set k of the KNN of every useful user you want to find

k=20

train_data = train_clean_data.collect()
users = user_array.collect()


for user in useful_user_array.toLocalIterator():


    KNN = findKNN(k,user,train_data,users)
    break
KNN








user1 = train_clean_data.filter(lambda x: x[0]==4)
user1.take(100)
user2 = train_clean_data.filter(lambda x: x[0]==687)
user2.take(100)

#begin test
us1=user1.collect()
us2=user2.collect()

us2[1]
us2[1][1]

len(us1)

zaza = 0
for i in range(len(us1)):
    for j in range(len(us2)):
        if us1[i][1]==us2[j][1]:

            zaza=zaza + math.pow((us1[i][2]-us2[j][2]),2)

zaza
#end test
