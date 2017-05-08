from pyspark import SparkContext
import csv
from pyspark.mllib.linalg import Matrix, Matrices
from scipy import linalg
import numpy as np
sc = SparkContext.getOrCreate()




train_rdd = sc.textFile("data/train.csv")
icm_rdd = sc.textFile("data/icm.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()

print(train_header)
print(train_rdd.count())
print(icm_header)
print(icm_rdd.count())

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))

user_array = train_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False)
item_array= icm_clean_data.map( lambda x: int(x[0])).sortBy(lambda x: x, ascending=False)
features_array = icm_clean_data.map( lambda x: int(x[1])).sortBy(lambda x: x, ascending=False)

U_dimension=user_array.take(1)[0]+2
I_dimension=item_array.take(1)[0]+2
F_dimension=features_array.take(1)[0]+2
U_dimension
I_dimension
F_dimension


UxI_weighted=np.zeros(shape=(U_dimension,I_dimension))
UxI_one=np.zeros(shape=(U_dimension,I_dimension))
IxF=np.zeros(shape=(I_dimension,F_dimension))


for user in train_clean_data.toLocalIterator():
    UxI_weighted[user[0]][user[1]]=user[2]
    UxI_weighted[user[0]][user[1]]=1

UxF=np.dot(UxI_weighted,IxF)
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
