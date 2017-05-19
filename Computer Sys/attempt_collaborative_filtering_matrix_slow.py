from pyspark import SparkContext
from scipy import sparse as sm
from sklearn.preprocessing import normalize
import numpy as np
import csv
#from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp
import sklearn.preprocessing as pp
sc = SparkContext.getOrCreate()

train_rdd = sc.textFile("data/train.csv")
icm_rdd = sc.textFile("data/icm.csv")
test_rdd= sc.textFile("data/target_users.csv")

train_header = train_rdd.first()
icm_header = icm_rdd.first()
test_header= test_rdd.first()

train_clean_data = train_rdd.filter(lambda x: x != train_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
icm_clean_data = icm_rdd.filter(lambda x: x != icm_header).map(lambda line: line.split(',')).map(lambda x: (int(x[0]), int(x[1])))
test_clean_data= test_rdd.filter(lambda x: x != test_header).map(lambda line: line.split(','))

test_users=test_clean_data.map( lambda x: int(x[0])).collect()


grouped_rates = train_clean_data.filter(lambda x: x[0] in test_users).map(lambda x: (x[0],x[1])).groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
grouped_rates_dic = dict(grouped_rates)


item_ratings = train_clean_data.map(lambda x: (x[0], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))
user_ratings_mean = item_ratings.mapValues(lambda x: (x[0] / (x[1]))).collect()
user_ratings_mean_dic=dict(user_ratings_mean)


item_ratings_forTop = train_clean_data.map(lambda x: (x[1], x[2])).aggregateByKey((0,0), lambda x,y: (x[0] + y, x[1] + 1),lambda x,y: (x[0] + y[0], x[1] + y[1]))#.sortBy(lambda x: x[1][1], ascending=False)
#item_ratings.take(10)
shrinkage_factor = 20
item_ratings_mean = item_ratings_forTop.mapValues(lambda x: (x[0] / (x[1] + shrinkage_factor))).sortBy(lambda x: x[1], ascending = False).map(lambda x: x[0]).collect()


users = train_clean_data.map(lambda x: x[0]).collect()
items = train_clean_data.map(lambda x: x[1]).collect()
ratings = train_clean_data.map(lambda x: x[2]-user_ratings_mean_dic[x[0]]).collect()

items_for_features= icm_clean_data.map(lambda x:x[0]).collect()
features = icm_clean_data.map(lambda x:x[1]).collect()

unos=[1]*len(items_for_features)

UxI= sm.csr_matrix((ratings, (users, items)))
IxF= sm.csr_matrix((unos, (items_for_features, features)))


def calcden(a,b, index1, index2):
    denpt1=0
    denpt2=0
    terms = set(a).intersection(b)
    for t in terms:
        denpt1+=np.power(UxI_lil[index1, t],2)
        denpt2+=np.power(UxI_lil[index2, t],2)
    return np.sqrt(denpt1)*np.sqrt(denpt2)+4


'''matrici di test'''
UxI_prep=[[2.5,-1.5,0,-0.5,-0.5],
         [-2.6,1.4,-1.6,1.4,1.4],
         [-1.5,0,-0.5,1.5,0.5],
         [0.25,-0.75,1.25,-0.75,0]]

UxI=sm.csr_matrix(UxI_prep)
'''end of test'''






''' -----------------------------tipo 1 item based'''





'''calcolo manuale delle similarità'''
IxI_sim=UxI.T.dot(UxI)
UxI_lil1=UxI.tolil()
UxI_lil=UxI_lil1.T
IxI_sim_lil=IxI_sim.tolil()
nritem=IxI_sim.shape[0]

nritem
teta=0
for i in range(nritem):
    for j in range(nritem):
        indexes_i = UxI_lil.getrow(i).nonzero()[1]
        indexes_j = UxI_lil.getrow(j).nonzero()[1]
        den=(calcden(indexes_i,indexes_j,i,j))
        if den!=0:
            IxI_sim_lil[i,j]/=den
    teta+=1
    print(teta)
    #if teta==10:
        #break

'''calcolo delle predictions'''
IxI_sim=IxI_sim_lil.tocsr()
IxI_sim.setdiag(0)
UxI_pred=UxI.dot(IxI_sim)

'''controllo'''
IxI_sim.toarray()
UxI_pred.toarray()








'''-------------------------------------------------tipo 2 user based'''

'''prendo similarità tra utenti già calcolate con rddd'''
UxU_sim_dafile=sc.textFile("users-users-sim.csv").map(lambda x: x.replace("(","").replace(")","").replace(" ","").split(",")).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
us1=UxU_sim_dafile.map(lambda x:x[0]).collect()
us2=UxU_sim_dafile.map(lambda x:x[1]).collect()
sims=UxU_sim_dafile.map(lambda x:x[2]).collect()
UxU_sim= sm.csr_matrix((sims, (us1, us2)))


'''calcolo manuale similarità tra utenti'''
UxU_sim=UxI.dot(UxI.T)
UxI_lil=UxI.tolil()
UxU_sim_lil=UxU_sim.tolil()
nruser=UxU_sim.shape[0]
nruser
teta=0
for i in range(nruser):
    for j in range(nruser):
        indexes_i = UxI_lil.getrow(i).nonzero()[1]
        indexes_j = UxI_lil.getrow(j).nonzero()[1]
        den=(calcden(indexes_i,indexes_j,i,j))
        if den!=0:
            UxU_sim[i,j]/=den
    teta+=1
    print(teta)
    if teta==10:
        break


'''calcolo delle predictions'''
UxU_sim.setdiag(0)
UxI_pred=UxU_sim.dot(UxI)


'''controlli'''
UxU_sim.toarray()
UxI_pred.toarray()
'''
#test per verificare se il calcolo del fede funziona
grouped_rates_dic[8]
UxI_pred[5,2145]
tony=UxI_pred.getrow(8).argmax()
tony
UxI_pred[8,tony]
UxI_pred[8,tony]=-50
'''


#3 remake of content-based
UxF=UxI_pred.dot(IxF)
UxF_norm=normalize(UxF,axis=1)
UxI_pred=UxF.dot(IxF.T)


c=0
f = open('submission_collaborative_ub1.csv', 'wt')
writer = csv.writer(f)
writer.writerow(('userId','RecommendedItemIds'))
for user in test_users:
    top=[0,0,0,0,0]

    user_predictions=UxI_pred.getrow(user)
    iterator = 0
    for i in range(5):
        prediction = user_predictions.argmax()
        while prediction in grouped_rates_dic[user] and prediction != 0:
            user_predictions[0,prediction]=-9
            prediction=user_predictions.argmax()
        if prediction == 0:
            prediction = item_ratings_mean[iterator]
            while prediction in grouped_rates_dic[user] or prediction in top:
                iterator += 1
                prediction = item_ratings_mean[iterator]
            iterator += 1
        else:
            user_predictions[0,prediction]=-9
        top[i]=prediction
    c+=1
    print(c)
    writer.writerow((user, '{0} {1} {2} {3} {4}'.format(top[0], top[1], top[2], top[3], top[4])))

f.close()



a=UxI.getrow(0).toarray()[0]
b=UxI.getrow(1).toarray()[0]
a
b
num=0
denpt1=0
denpt2=0
for i in range(len(a)):
    if a[i]!=0 and b[i]!=0:

        num+=a[i]*b[i]
        denpt1+=np.power(a[i],2)
        denpt2+=np.power(b[i],2)

sim=num/(np.sqrt(denpt1)*np.sqrt(denpt2))
sim
