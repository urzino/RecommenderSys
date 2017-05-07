from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.linalg import Matrix, Matrices
import csv
import operator

def countDistinct(rdd):
    return rdd.distinct().count()

def countSameFeatures(item1,item2):
    intersection = set(item1[1]).intersection(item2[1])
    return len(intersection)

sc = SparkContext.getOrCreate()

icm_rdd = sc.textFile("data/icm.csv")

header = icm_rdd.first()

print(header)
print(icm_rdd.count())

clean_data = icm_rdd.filter(lambda x: x != header).map(lambda line: line.split(','))#.filter(lambda x: float(x[2]) >= 8)

print(clean_data.take(10))

features = clean_data.map(lambda x: int(x[1]))

items = clean_data.map(lambda x: int(x[0]))

dist_features=features.distinct()
print("{0} distinct items and {1} distinct features".format(countDistinct(items), countDistinct(features)))

item_features = clean_data.map(lambda x: (int(x[0]),int(x[1]))).groupByKey().map(lambda x: (x[0], list(x[1])))
item_features.take(10)


f = open('KNN_ITEMS.csv', 'wt')
i=0;
try:
    writer = csv.writer(f)
    writer.writerow(('itemId','neighborId','weight'))
    for itm1 in item_features.toLocalIterator():
        dict={}
        for itm2 in item_features.toLocalIterator():

            if itm1[0]!=itm2[0]:
                dict[itm2[0]]= countSameFeatures(itm1,itm2)
        dict=sorted(dict.items(),key=operator.itemgetter(1),reverse=True)
        i=1+1
        print(i/36797)


        for (id, weight) in dict:
            writer.writerow((itm1[0], id, weight))
finally:
    f.close()
