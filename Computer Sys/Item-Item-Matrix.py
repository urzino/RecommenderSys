from pyspark import SparkContext
import csv
from pyspark.mllib.linalg import Matrix, Matrices
from scipy import linalg
import numpy as np

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
useful_user_array=test_clean_data.map( lambda x: int(x[0]))

grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_features.take(10)

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
    i = 0
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
        i = i + 1
        print(i / useful_user_array.count())
finally:
    f.close()
#UxI_pred
#predictions
#valore massimo per riga
#np.argmax(UxI_pred, axis = 1)[0]
'''
UxI_weighted=np.zeros(shape=(U_dimension,I_dimension))
UxI_one=np.zeros(shape=(U_dimension,I_dimension))
IxF=np.zeros(shape=(I_dimension,F_dimension))


for user in train_clean_data.toLocalIterator():
    UxI_weighted[user[0]][user[1]]=user[2]
    UxI_one[user[0]][user[1]]=1

del train_clean_data

for item in icm_clean_data.toLocalIterator():
    IxF[item[0]][item[1]]=1

del icm_clean_data


UxF=np.dot(UxI_weighted,IxF)
UxF_counter=np.dot(UxI_one,IxF)

for row in 0:U_dimension:
    for col in 0:F_dimension:
        if UxF_counter[row][col]!=0:
            UxF[row][col]=UxF[row][col]/UxF_counter[row][col]
'''
train_clean_data.take(10)
icm_clean_data.take(10)

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

#matrice iniziale di USERSxITEMS con valori di ratings (numero_righe, numero_col, colonne con valori, righe con valori, valori)
user_item_rating_matrix = Matrices.sparse(total_users, total_items, item_array.collect(), user_array.collect(), rating_array.collect())
