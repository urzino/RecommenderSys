from pyspark import SparkContext
import csv
from pyspark.mllib.linalg import Matrix, Matrices

sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("data/train.csv")
icm_rdd = sc.textFile("data/icm.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()

print(train_header)
print(train_rdd.count())
print(icm_header)
print(icm_rdd.count())

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(','))
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(','))


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
rating_array = train_clean_data.map(lambda x: int(x[2]))
rating_array.take(10)

#tutti gli item anche se non sono stati votati
items = icm_clean_data.map(lambda x: int(x[0]))
total_items = items.distinct().count()
print(total_items)

#matrice iniziale di USERSxITEMS con valori di ratings (numero_righe, numero_col, colonne con valori, righe con valori, valori)
user_item_rating_matrix = Matrices.sparse(total_users, total_items, item_array.collect(), user_array.collect(), rating_array.collect())
