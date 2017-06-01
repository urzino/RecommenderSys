import numpy as np
from pyspark import SparkContext
from scipy.sparse import csc_matrix

#constants defining the dimensions of our User Rating Matrix (URM)
MAX_PID = 37143
MAX_UID = 15375

sc = SparkContext.getOrCreate()

def readUrm(train_clean_data):
	urm = np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)
	for row in train_clean_data:
		urm[row[0], row[1]] = row[2]

	return csc_matrix(urm, dtype=np.float32)

def readUsersTest(test_clean_data):
	uTest = dict()
	for row in test_clean_data:
		uTest[row] = list()

	return uTest

def getMoviesSeen(train_clean_data):
	moviesSeen = dict()
	for row in train_clean_data:
		try:
			moviesSeen[row[0]].append(row[1])
		except:
			moviesSeen[row[0]] = list()
			moviesSeen[row[0]].append(row[1])

	return moviesSeen

import math as mt
import csv
from sparsesvd import sparsesvd

def computeSVD(urm, K):
	U, s, Vt = sparsesvd(urm, K)

	dim = (len(s), len(s))
	S = np.zeros(dim, dtype=np.float32)
	for i in range(0, len(s)):
		S[i,i] = mt.sqrt(s[i])

	U = csr_matrix(np.transpose(U), dtype=np.float32)
	S = csr_matrix(S, dtype=np.float32)
	Vt = csr_matrix(Vt, dtype=np.float32)

	return U, S, Vt

from scipy.sparse.linalg import * #used for matrix multiplication

def computeEstimatedRatings(urm, U, S, Vt, uTest, moviesSeen, K, test):
	rightTerm = S*Vt

	estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
	for userTest in uTest:
		prod = U[userTest, :]*rightTerm

		estimatedRatings[userTest, :] = prod.todense()
		recom = (-estimatedRatings[userTest, :]).argsort()[:250]
		for r in recom:
			if r not in moviesSeen[userTest]:
				uTest[userTest].append(r)

				if len(uTest[userTest]) == 5:
					break

	return uTest

def printResult(uTest):
    f = open('submission_svd.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(('userId','RecommendedItemIds'))
    for u in uTest:
        predictions = uTest[u]
        iterator = 0
        for i in range(5 - len(predictions)):
            while (item_ratings_mean[iterator] in moviesSeen[u]) or (item_ratings_mean[iterator] in predictions):
                iterator = iterator + 1
            predictions = predictions + [item_ratings_mean[iterator]]
        writer.writerow((u, '{0} {1} {2} {3} {4}'.format(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])))

    f.close()

K = 90
train_rdd = sc.textFile("data/train.csv")
train_header = train_rdd.first()
train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
train_data = train_clean_data.collect()
test_rdd= sc.textFile("data/target_users.csv")
test_header= test_rdd.first()
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda x: int(x)).collect()
item_ratings = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
shrinkage_factor = 5
item_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()
urm = readUrm(train_data)
U, S, Vt = computeSVD(urm, K)
uTest = readUsersTest(test_clean_data)
moviesSeen = getMoviesSeen(train_data)
uTest = computeEstimatedRatings(urm, U, S, Vt, uTest, moviesSeen, K, True)
printResult(uTest)
