
import numpy as np
import pandas as pd

a=pd.read_csv('/Users/dean/Desktop/Web economics/newdata/train_index.csv')
b=pd.read_csv('/Users/dean/Desktop/Web economics/data/train.csv')
a["click"]=b["click"]


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random
import pickle

# A, paths
train='/Users/dean/Desktop/Web economics/newdata/train_index.csv'
test='/Users/dean/Desktop/Web economics/newdata/val_index.csv'#'vali_100.tsv'
submission = 'ftrltrain.csv'  # path of to be outputted submission file

# B, model
alpha = .05  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 10.     # L1 regularization, larger value means more regularized
L2 = 0.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
#holdafter = 9   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [random() for k in range(D)]#[0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path), delimiter=',')):
        # process id
        #print row
        
        try:
            #ID= row['Unnamed: 0']
            del row['Unnamed: 0']
            
        except:
            pass
        # process clicks
        y = 0.
        target='click'#'IsClick' 
        if target in row:
            if row[target] == '1':
                y = 1.
            del row[target]

        # extract date

        # turn hour really into hour, it was originally YYMMDDHH

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield x, y
start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# start training
for e in range(epoch):
    loss = 0.
    count = 0
    for x, y in data(train, D):  # data is a generator

        p = learner.predict(x)
        loss += logloss(p, y)
        learner.update(x, p, y)
        count+=1
        if count%1000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), count, loss/count))
        #if count>10000: # comment this out when you run it locally.
            #break
count=0
loss=0
#import pickle
#pickle.dump(learner,open('ftrl3.p','w'))
print ('write result')
##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
with open(submission, 'w') as outfile:
    outfile.write('target\n')
    for  x, y in data(train, D):
        count+=1
        p = learner.predict(x)
        loss += logloss(p, y)

        outfile.write('%s\n' % (str(p)))
        if count%1000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), count, loss/count))

with open(submission, 'w') as outfile:
    outfile.write('target\n')
    for  x, y in data(test, D):
        count+=1
        p = learner.predict(x)
        loss += logloss(p, y)

        outfile.write('%s\n' % (str(p)))
        if count%1000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), count, loss/count))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import math
import seaborn as sns


from sklearn.metrics import roc_auc_score
a=pd.read_csv('/Users/dean/Desktop/Web economics/newdata/val_index.csv')
b =pd.read_csv('/Users/dean/Downloads/ftrl1sub.csv')
roc_auc_score(a["click"],b["target"])

from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(a["click"],b["target"]))

b=pd.read_csv('/Users/dean/Desktop/Web economics/newdata/val_index.csv')
del b["Unnamed: 0"]

f = b["click"]
del b["click"]
sum(xgbmodel.predict(b)==0)
sum(xgbmodel.predict(b)==0)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
NB_Prob = []
clf.predict_proba(b)
NB_Prob = []
NB_raw = clf.predict_proba(b)
for i in NB_raw:
    NB_Prob.append(i[1])
len(NB_Prob)
len(f)
from sklearn.metrics import roc_auc_score
roc_auc_score(f,NB_Prob)
v =pd.read_csv('/Users/dean/Downloads/ftrl1sub.csv')
list(v["target"])

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(list(w["target"]),bins=1000) #bins=100
plt.xlabel("predicted CTR of FTRL Proximal model")
plt.ylabel("count")
#plt.axvline(x=validation_avg_ctr, color='r')
plt.xlim(0, 0.01)
plt.show()
#plt.savefig('FTRL_Dist_Train.png',dpi=400)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(list(NB_Prob),bins=1000) #bins=100
plt.xlabel("predicted CTR of FTRL Proximal model")
plt.ylabel("count")
#plt.axvline(x=validation_avg_ctr, color='r')
plt.xlim(0, 0.01)
#plt.show()
plt.savefig('NB_DISTR.png',dpi=400)

from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(f,NB_Prob))


from sklearn.metrics import roc_curve
fpr_NB, tpr_NB, TH = roc_curve(f, NB_Prob)
fpr_FTRL, tpr_FTRL, THF = roc_curve(f, list(v["target"]))

plt.plot(fpr_NB, tpr_NB, color='C0',label='Naive Bayes ROC' )
plt.plot(fpr_FTRL,tpr_FTRL,color='C1',label='FTRL Proximal ROC')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc=4)
#plt.show()

#plt.tight_layout()
plt.savefig('ROC_Mingda.png',dpi=400)


validation=pd.read_csv("/Users/dean/Desktop/Web economics/newdata/val_index.csv")
l =['IP', 'adexchange', 'creative', 'domain', 'keypage',
       'slotformat', 'slotheight', 'slotid', 'slotvisibility', 'slotwidth',
       'url', 'useragent', 'userid', 'tag0', 'tag1', 'tag2', 'tag3', 'tag4',
       'tag5', 'tag6', 'tag7', 'tag8', 'tag9', 'tag10', 'tag11', 'tag12',
       'tag13', 'tag14', 'tag15', 'tag16', 'tag17', 'tag18', 'tag19', 'tag20',
       'tag21', 'tag22', 'tag23', 'tag24', 'tag25', 'tag26', 'tag27', 'tag28',
       'tag29', 'tag30', 'tag31', 'tag32', 'tag33', 'tag34', 'tag35', 'tag36',
       'tag37', 'tag38', 'tag39', 'tag40', 'tag41', 'tag42', 'tag43', 'tag44',
       'tag45', 'tag46', 'tag47', 'tag48', 'tag49', 'tag50', 'tag51', 'tag52',
       'tag53', 'tag54', 'tag55', 'tag56', 'tag57', 'tag58', 'tag59', 'tag60',
       'tag61', 'tag62', 'tag63', 'tag64', 'tag65', 'tag66', 'tag67', 'tag68',
        'weekday', 'hour', 'region', 'city', 'slotprice', 'bidprice',
       'payprice', 'advertiser']
validation_data=validation[l]
validation_click=validation['click']
validation_click.value_counts()
val_avgCTR=sum(validation['click'])/len(validation)
val_avgCTR

NB_6250_Lin=[validation_budget(i * (NB_Prob/val_avgCTR),6250) for i in range (0,300)]
NB_6250_Lin_clicks = []
for i in range(0,300):
    NB_6250_Lin_clicks.append(NB_6250_Lin[i][4])
FTRL_6250_Lin=[validation_budget(i * (list(v["target"])/val_avgCTR),6250) for i in range (0,300)]#
FTRL_6250_Lin_clicks = []
for i in range(0,300):
    FTRL_6250_Lin_clicks.append(FTRL_6250_Lin[i][4])
#plt.plot(FTRL_6250_Lin_clicks)

plt.plot(FTRL_6250_Lin_clicks)
plt.show()
best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_2(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,6250))
FTRL_NON_2_6250 = best

FTRL_NON_2_6250 = best
FTRL_6250_NON2_clicks = []
for i in range(0,299):
    FTRL_6250_NON2_clicks.append(FTRL_NON_2_6250[i][4])

plt.plot(FTRL_6250_NON2_clicks)
plt.show()


best_1 = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_1(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best_1.append(validation_budget(bid_price_nonlinear,6250))
FTRL_NON_1_6250 = best_1

FTRL_6250_NON1_clicks = []
for i in range(0,299):
    FTRL_6250_NON1_clicks.append(FTRL_NON_1_6250[i][4])
plt.plot(FTRL_6250_NON1_clicks)
plt.show()

def validation_budget(bidding,budget):
    new= validation.assign(bidding = bidding, diff= bidding -validation['payprice'])
    new= new[new['diff'] >= 0]
    cost= 0
    index= 0
    for pay in new.payprice:
        cost= pay + cost
        index += 1
        if cost/1000 >= budget:
            break
        
    
    clicks =sum(new['click'][:index])
    ctr= clicks/index
    spend = sum(new['payprice'][:index])/1000
    cpc = spend/clicks
    cpm = spend/index
    return [ctr,spend,cpc,cpm,clicks]

def ORTB_2(theta,c): # theta is the ctr, c is a constant
    lam=5.2*10**(-7) #lambda
    #optimal_bid_price=c*(((theta + np.sqrt(c**2*lam**2 + theta**2))/(c*lam))**(1/3)-((c*lam)/(theta + np.sqrt((c**2*(lam**2) + theta**2)))^(1/3))
    optimal_bid_price=c*(((theta + np.sqrt(c**2*lam**2 + theta**2))/(c*lam))**(1/3)-((c*lam)/(theta + np.sqrt((c**2*(lam**2) + theta**2)))**(1/3)))
    return optimal_bid_price
def ORTB_1(theta,c): # theta is the ctr, c is a constant
    lam=5.2*10**(-7) #lambda
    #optimal_bid_price=c*(((theta + np.sqrt(c**2*lam**2 + theta**2))/(c*lam))**(1/3)-((c*lam)/(theta + np.sqrt((c**2*(lam**2) + theta**2)))^(1/3))
    #optimal_bid_price=c*(((theta + np.sqrt(c**2*lam**2 + theta**2))/(c*lam))**(1/3)-((c*lam)/(theta + np.sqrt((c**2*(lam**2) + theta**2)))**(1/3)))
    optimal_bid_price = (np.sqrt((c*theta/lam)+c**2))-c
    return optimal_bid_price
plt.plot( FTRL_6250_Lin_clicks, color='C0',label='Linear bidding' )
plt.plot(FTRL_6250_NON1_clicks,color='C1',label='ORTB_1')
plt.plot(FTRL_6250_NON2_clicks,color='C2',label='ORTB_2')

plt.xlabel('Base Bid/C value')
plt.ylabel('Clicks')
plt.legend(loc=4)
#plt.show()

#plt.tight_layout()
plt.savefig('FTRL_BID_CLICKS.png',dpi=400)

FTRL_6250_NON1_CTR = []
for i in range(0,299):
    FTRL_6250_NON1_CTR.append(FTRL_NON_1_6250[i][0])

FTRL_6250_NON1_Spend = []
for i in range(0,299):
    FTRL_6250_NON1_Spend.append(FTRL_NON_1_6250[i][1])

FTRL_6250_NON1_CPC = []
for i in range(0,299):
    FTRL_6250_NON1_CPC.append(FTRL_NON_1_6250[i][2])

FTRL_6250_NON1_CPM = []
for i in range(0,299):
    FTRL_6250_NON1_CPM.append(FTRL_NON_1_6250[i][3])

FTRL_6250_NON2_CTR = []
for i in range(0,299):
    FTRL_6250_NON2_CTR.append(FTRL_NON_2_6250[i][0])

FTRL_6250_NON2_Spend = []
for i in range(0,299):
    FTRL_6250_NON2_Spend.append(FTRL_NON_2_6250[i][1])

FTRL_6250_NON2_CPC = []
for i in range(0,299):
    FTRL_6250_NON2_CPC.append(FTRL_NON_2_6250[i][2])

FTRL_6250_NON2_CPM = []
for i in range(0,299):
    FTRL_6250_NON2_CPM.append(FTRL_NON_2_6250[i][3])

FTRL_6250_Lin_CTR = []
for i in range(0,300):
    FTRL_6250_Lin_CTR.append(FTRL_6250_Lin[i][0])

FTRL_6250_Lin_Spend = []
for i in range(0,300):
    FTRL_6250_Lin_Spend.append(FTRL_6250_Lin[i][1])
FTRL_6250_Lin_CPC = []
for i in range(0,300):
    FTRL_6250_Lin_CPC.append(FTRL_6250_Lin[i][2])
FTRL_6250_Lin_CPM = []
for i in range(0,300):
    FTRL_6250_Lin_CPM.append(FTRL_6250_Lin[i][3])
FTRL_6250_Lin_CTR = []
for i in range(0,300):
    FTRL_6250_Lin_CTR.append(FTRL_6250_Lin[i][0])
FTRL_6250_Lin_CTR[65]
plt.plot( FTRL_6250_Lin_CTR, color='C0',label='Linear bidding' )
plt.plot(FTRL_6250_NON1_CTR,color='C1',label='ORTB_1')
plt.plot(FTRL_6250_NON2_CTR,color='C2',label='ORTB_2')

plt.xlabel('Base Bid/C value')
plt.ylabel('CTR')
plt.legend(loc=4)
#plt.show()

#plt.tight_layout()
plt.savefig('FTRL_BID_CTR.png',dpi=400)
plt.plot( FTRL_6250_Lin_CPC, color='C0',label='Linear bidding' )
plt.plot(FTRL_6250_NON1_CPC,color='C1',label='ORTB_1')
plt.plot(FTRL_6250_NON2_CPC,color='C2',label='ORTB_2')

plt.xlabel('Base Bid/C value')
plt.ylabel('CPC')
plt.legend(loc=4)
#plt.show()

#plt.tight_layout()
plt.savefig('FTRL_BID_CPC.png',dpi=400)
plt.plot( FTRL_6250_Lin_Spend, color='C0',label='Linear bidding' )
plt.plot(FTRL_6250_NON1_Spend,color='C1',label='ORTB_1')
plt.plot(FTRL_6250_NON2_Spend,color='C2',label='ORTB_2')

plt.xlabel('Base Bid/C value')
plt.ylabel('Spend')
plt.legend(loc=4)
#plt.show()

#plt.tight_layout()
plt.savefig('FTRL_BID_Spend.png',dpi=400)
plt.plot( FTRL_6250_Lin_CPM, color='C0',label='Linear bidding' )
plt.plot(FTRL_6250_NON1_CPM,color='C1',label='ORTB_1')
plt.plot(FTRL_6250_NON2_CPM,color='C2',label='ORTB_2')

plt.xlabel('Base Bid/C value')
plt.ylabel('CPM')
plt.legend(loc=4)
#plt.show()

#plt.tight_layout()
plt.savefig('FTRL_BID_CPM.png',dpi=400)
FTRL_6250_Lin_Spend
FTRL_12500_Lin=[validation_budget(i * (list(v["target"])/val_avgCTR),12500) for i in range (0,300)]#

FTRL_12500_Lin_clicks = []
for i in range(0,300):
    FTRL_12500_Lin_clicks.append(FTRL_12500_Lin[i][4])

FTRL_12500_Lin_CTR = []
for i in range(0,300):
    FTRL_12500_Lin_CTR.append(FTRL_12500_Lin[i][0])

FTRL_12500_Lin_CTR[118]

FTRL_3125_Lin=[validation_budget(i * (list(v["target"])/val_avgCTR),3125) for i in range (0,300)]#

FTRL_3125_Lin_clicks = []
for i in range(0,300):
    FTRL_3125_Lin_clicks.append(FTRL_3125_Lin[i][4])

FTRL_3125_Lin_CTR = []
for i in range(0,300):
    FTRL_3125_Lin_CTR.append(FTRL_3125_Lin[i][0])

max(FTRL_3125_Lin_clicks)
FTRL_3125_Lin_clicks.index(134)
FTRL_3125_Lin_CTR[42]
FTRL_25000_Lin=[validation_budget(i * (list(v["target"])/val_avgCTR),25000) for i in range (0,300)]#
FTRL_25000_Lin_clicks = []
for i in range(0,300):
    FTRL_25000_Lin_clicks.append(FTRL_25000_Lin[i][4])

FTRL_25000_Lin_CTR = []
for i in range(0,300):
    FTRL_25000_Lin_CTR.append(FTRL_25000_Lin[i][0])
max( FTRL_25000_Lin_clicks)
FTRL_25000_Lin_clicks.index(217)
FTRL_25000_Lin_CTR[295]

FTRL_6250_Lin_CTR[65]
best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_2(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,25000))
FTRL_NON_2_25000 = best
FTRL_25000_NON2_clicks = []
for i in range(0,299):
    FTRL_25000_NON2_clicks.append(FTRL_NON_2_25000[i][4])
FTRL_25000_NON2_CTR = []
for i in range(0,299):
    FTRL_25000_NON2_CTR.append(FTRL_NON_2_25000[i][0])

best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_2(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,12500))
FTRL_NON_2_12500 = best
FTRL_12500_NON2_clicks = []
for i in range(0,199):
    FTRL_12500_NON2_clicks.append(FTRL_NON_2_12500[i][4])
FTRL_12500_NON2_CTR = []
for i in range(0,199):
    FTRL_12500_NON2_CTR.append(FTRL_NON_2_12500[i][0])

max(FTRL_12500_NON2_clicks)
FTRL_12500_NON2_CTR[23]


best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_2(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,3125))
FTRL_NON_2_3125 = best
FTRL_3125_NON2_clicks = []
for i in range(0,299):
    FTRL_3125_NON2_clicks.append(FTRL_NON_2_3125[i][4])
FTRL_3125_NON2_CTR = []
for i in range(0,299):
    FTRL_3125_NON2_CTR.append(FTRL_NON_2_3125[i][0])
max(FTRL_3125_NON2_clicks)
FTRL_3125_NON2_clicks.index(91)
FTRL_3125_NON2_CTR[6]

best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_1(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,25000))
FTRL_NON_1_25000 = best
FTRL_25000_NON1_clicks = []
for i in range(0,299):
    FTRL_25000_NON1_clicks.append(FTRL_NON_1_25000[i][4])
FTRL_25000_NON1_CTR = []
for i in range(0,299):
    FTRL_25000_NON1_CTR.append(FTRL_NON_1_25000[i][0])
max(FTRL_25000_NON1_clicks)
FTRL_25000_NON1_clicks.index(222)
FTRL_25000_NON1_CTR[273]
best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_1(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,12500))
FTRL_NON_1_12500 = best
FTRL_12500_NON1_clicks = []
for i in range(0,299):
    FTRL_12500_NON1_clicks.append(FTRL_NON_1_12500[i][4])
FTRL_12500_NON1_CTR = []
for i in range(0,299):
    FTRL_12500_NON1_CTR.append(FTRL_NON_1_12500[i][0])

best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_1(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,12500))

FTRL_NON_1_12500 = best
FTRL_12500_NON1_clicks = []
for i in range(0,299):
    FTRL_12500_NON1_clicks.append(FTRL_NON_1_12500[i][4])
FTRL_12500_NON1_CTR = []
for i in range(0,299):
    FTRL_12500_NON1_CTR.append(FTRL_NON_1_12500[i][0])
max(FTRL_12500_NON1_clicks)
FTRL_12500_NON1_clicks.index(187)
FTRL_12500_NON1_CTR[11]


best = []
for c in range(1,300,1):
   
    bid_price_nonlinear=[]
    for b in list(v["target"]):
        
        bid_price_nonlinear.append(ORTB_1(b,c))
    #validation_budget(bid_price_nonlinear,6250)
    best.append(validation_budget(bid_price_nonlinear,3125))
FTRL_NON_1_3125 = best
FTRL_3125_NON1_clicks = []
for i in range(0,299):
    FTRL_3125_NON1_clicks.append(FTRL_NON_1_3125[i][4])
FTRL_3125_NON1_CTR = []
for i in range(0,299):
    FTRL_3125_NON1_CTR.append(FTRL_NON_1_3125[i][0])

max(FTRL_3125_NON1_clicks)
FTRL_3125_NON1_clicks.index(106)
FTRL_3125_NON1_CTR[1]


