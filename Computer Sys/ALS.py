from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import csv
#from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext.getOrCreate()

icm_rdd = sc.textFile("data/train.csv")

header = icm_rdd.first()

print(header)
print(icm_rdd.count())

clean_data = icm_rdd.filter(lambda x: x != header).map(lambda line: line.split(','))#.filter(lambda x: float(x[2]) >= 8)

print(clean_data.take(10))

rate = clean_data.map(lambda x: int(x[2]))
print('got', rate.count(), 'ratings')
#print(rate.mean())

users = clean_data.map(lambda x: int(x[0]))

print('from', users.distinct().count(), 'distinct users')

items = clean_data.map(lambda x: int(x[1]))

print('for', items.distinct().count(), 'distinct items')

#Rating(user, item, rating)
ratings = clean_data.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))

#user_ratings = ratings.keyBy(lambda x : x[0]).lookup(2738)

#print(user_ratings.size)

rank = 20
num_iterations = 20

samples = ratings.take(10)
print('10 random ratings:')
for sample in samples:
    print(sample)

ratings.cache()

#ALS

sc.setCheckpointDir('checkpoint/')
ALS.checkpointInterval = 2
model = ALS.train(ratings, rank, num_iterations, seed=1234)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

#train + test
'''
train, test = ratings.randomSplit([0.7,0.3],7856)
train.cache()
test.cache()

model = ALS.train(train, rank, num_iterations)

testdata = train.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

testdata2 = test.map(lambda p: (p[0], p[1]))
predictions2 = model.predictAll(testdata2).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds2 = test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions2)
MSE2 = ratesAndPreds2.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE2))
'''
print(model.productFeatures().count())

print(model.userFeatures().count())

#predictions = model.transform(test)
#evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
#rmse = evaluator.evaluate(predictions)
#print("Root-mean-square error = " + str(rmse))

#for product X, finds N users
#print(model.recommendUsers(1, 5))

#for user X, finds N products
top5 = model.recommendProducts(7393, 5)
top5[0].rating
#top N for every user
#top5 = model.recommendProductsForUsers(5)

for top in top5:
    print(top)

#single user for single product
print(model.predict(7393, 90))

#model.save(sc, "first_model")

test_users = sc.textFile("data/test.csv")

test_header = test_users.first()

clean_users = test_users.filter(lambda x: x != test_header).map(lambda y: int(y))
clean_users.count()
clean_users.take(10)

clean_users.cache()

f = open('submission.csv', 'wt')
try:
    writer = csv.writer(f)
    writer.writerow(('userId','RecommendedItemIds'))
    for user in clean_users.toLocalIterator():
        top5_test = model.recommendProducts(user, 5)
        writer.writerow((user, '{0} {1} {2} {3} {4}'.format(top5_test[0].rating, top5_test[1].rating, top5_test[2].rating, top5_test[3].rating, top5_test[4].rating)))
        for top in top5_test:
            print(top)
finally:
    f.close()
