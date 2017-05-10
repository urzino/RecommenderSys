from pyspark import SparkContext
import csv
#from pyspark.mllib.linalg import Matrix, Matrices
from scipy import linalg, sparse
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
#test_users.take(10)

#for every item all its features
grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_features.take(10)
grouped_features.cache()

#for every features all its items
grouped_items = icm_clean_data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_items.take(10)
grouped_items.cache()

#for every user all its ratings (item, rate)
grouped_rates = train_clean_data.map(lambda x: (x[0],(x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_rates.take(10)
grouped_rates.cache()

#return only test users
def is_in_test(user):
    return user[0] in test_users

test_user_ratings = grouped_rates.filter(is_in_test)
test_user_ratings.take(10)
test_user_ratings.cache()

#returns all the features voted by the user
def calculate_features_ratings(user_rates):
    user = user_rates[0]
    item_rates = dict(user_rates[1])
    #features_rates = list()
    #for i in range(len(item_rates)):
    temp = grouped_features.filter(lambda x: item_rates.get(x[0], -1) != -1)

temp2 = grouped_rates.take(1)[0][1]
temp3 = list(map(lambda x: x[0], temp2))
temp3
temp = dict(temp2)
y = grouped_features.filter(lambda x: temp.get(x[0], -1) != -1).flatMap(lambda x: [(f, temp[x[0]]) for f in x[1]])
temp
#[item for item in temp if item[0] == 1][0]
y.distinct().count()

user_features_ratings = grouped_rates.map(calculate_features_ratings)

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
