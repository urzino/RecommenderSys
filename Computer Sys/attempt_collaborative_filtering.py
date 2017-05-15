from pyspark import SparkContext
from scipy import sparse as sm
import numpy as np

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

item_ratings = train_clean_data.map(lambda x: (x[0], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
user_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1]))).collect()
user_ratings_mean_dic=dict(user_ratings_mean)


users = train_clean_data.map(lambda x: x[0]).collect()
items = train_clean_data.map(lambda x: x[1]).collect()
ratings = train_clean_data.map(lambda x: x[2]-user_ratings_mean_dic[x[0]]).collect()



UxI= sm.csr_matrix((ratings, (users, items)))

IxI_sim=UxI.transpose().dot(UxI)






'''items_distinct = train_clean_data.map(lambda x: x[1]).distinct().collect()
max_item= max(items_distinct)
max_item
IxI_sim = sm.lil_matrix((max_item+1,max_item+1))'''

'''
item_ratings = train_clean_data.map(lambda x: (x[0], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
user_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1]))).collect()
user_ratings_mean_dic=dict(user_ratings_mean)
max_user= UxI.shape[0]
max_user
for item1 in items_distinct:
    for item2 in items_distinct:
        if item1!=item2:
            Col_item1=UxI.getcol(item1)
            Col_item2=UxI.getcol(item2)
            Numerator=0
            Denominator_part1=0
            Denominator_part2=0
            for i in range(max_user):
                if (Col_item1[i,0]!=0 and Col_item2[i,0]!=0):
                    Num_part1=Col_item1[i,0] - user_ratings_mean_dic[i]
                    Num_part2=Col_item2[i,0] - user_ratings_mean_dic[i]
                    Numerator += Num_part1*Num_part2
                    Denominator_part1 += np.power(Num_part1,2)
                    Denominator_part2 += np.power(Num_part2,2)
            IxI_sim[item1,item2]=Numerator/(np.sqrt(Denominator_part1)*np.sqrt(Denominator_part2))
    break

IxU=U.I.transpose'''
