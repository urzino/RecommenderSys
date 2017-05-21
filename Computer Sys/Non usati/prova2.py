from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
#from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext.getOrCreate()

def countDistinct(rdd):
    return rdd.distinct().count()

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

items = icm_clean_data.map(lambda x: int(x[0]))
features = icm_clean_data.map(lambda x: int(x[1]))
print("{0} distinct items and {1} distinct features".format(countDistinct(items), countDistinct(features)))

users = train_clean_data.map(lambda x: int(x[0]))
train_items = train_clean_data.map(lambda x: int(x[1]))
rates = train_clean_data.map(lambda x: int(x[2]))
print("{0} distinct users and {1} distinct items and {2} rates".format(countDistinct(users), countDistinct(train_items), rates.count()))
print(train_clean_data.count() / (users.distinct().count() * train_items.distinct().count()) * 100)
grouped_rates = train_clean_data.map(lambda x: (x[0],(x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_rates.take(10)

grouped_features = icm_clean_data.map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1])))
grouped_features.take(10)
