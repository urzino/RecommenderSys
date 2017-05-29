from pyspark import SparkContext
from algorithms.content_fede_funzionante import calculate_content_matrix
from algorithms.user_user_fede import calculate_user_collaborative_matrix
import csv

sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("data/train.csv")
icm_rdd = sc.textFile("data/icm_fede.csv")
test_rdd= sc.textFile("data/target_users.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2]))).cache()
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]))).cache()
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(',')).cache()

content_fede = calculate_content_matrix(train_clean_data, icm_clean_data, test_clean_data)
collaborative_user = calculate_user_collaborative_matrix(train_clean_data, test_clean_data)
