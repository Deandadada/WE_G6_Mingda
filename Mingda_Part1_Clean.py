import pandas as pd
import numpy as np
%matplotlib inline
from sklearn import metrics
from random import randint

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

train_data= pd.read_csv("/Users/dean/Desktop/Web economics/data/train.csv")
test_data= pd.read_csv("/Users/dean/Desktop/Web economics/data/test.csv")
validation_data= pd.read_csv("/Users/dean/Desktop/Web economics/data/validation.csv")

new_train = train_data.advertiser.unique()
table = pd.DataFrame(columns=['advertiser', 'num_imp', 'num_clicks', 'costs', 'CTR', 'avgCPM', 'eCPC'])
for i in range(len(new_train)):
    adver = sta_train[i]
    df = train_data[train_data.advertiser == adver]
    clicks = df.click.sum()
    imps = df.bidid.count()
    costs = df.payprice.sum() / 1000
    CTR = ((clicks / imps) * 100)
    avgCPM = ((costs / imps) * 1000)
    eCPC = (costs / clicks)
    table.loc[i] = [adver, imps, clicks, costs, CTR, avgCPM, eCPC]

print(table)


advs = [3427, 2997, 3386, 3476, 2821,1458,3358,2261,2259]
dataframes = []
for new_train in advs:
    df = train_data[train_data['advertiser'] == new_train]
    ctr = df.groupby('weekday').agg({'click':{'click':sum}, 'bidid': {'imps' : 'count'}})
    ctr.columns = ctr.columns.droplevel(0)
    ctr['ctr'] = (ctr.click / ctr.imps) * 100
    dataframes.append(ctr)
plt.plot(dataframes[0]['ctr'], c='C0',label='user:'+str(advs[0]), marker='*')
plt.plot(dataframes[1]['ctr'], c='C1',label='user:'+str(advs[1]), marker='o')
plt.plot(dataframes[2]['ctr'], c='C2',label='user:'+str(advs[2]), marker='.')
plt.plot(dataframes[3]['ctr'], c='C3',label='user:'+str(advs[3]), marker='>')
plt.plot(dataframes[4]['ctr'], c='C4',label='user:'+str(advs[4]), marker='<')
plt.plot(dataframes[5]['ctr'], c='C5',label='user:'+str(advs[5]), marker='>')
plt.plot(dataframes[6]['ctr'], c='C6',label='user:'+str(advs[6]), marker='s')
plt.plot(dataframes[7]['ctr'], c='C7',label='user:'+str(advs[7]), marker='+')
plt.plot(dataframes[8]['ctr'], c='C8',label='user:'+str(advs[8]), marker='*')
#plt.plot(dataframes[9]['ctr'], c='m',label='user'+str(advs[9]), marker='>')

plt.ylabel('CTR (click through rate)')
plt.xlabel('Weekday')
plt.margins(0.05)
plt.legend()
#plt.show()
plt.savefig('CTR_ALL_WEEK.png',dpi=400)


dataframes = []
for adv in advs:
    df = train_data[train_data['advertiser'] == adv]
    ctr = df.groupby('hour').agg({'click':{'click':sum}, 'bidid': {'imps' : 'count'}})
    ctr.columns = ctr.columns.droplevel(0)
    ctr['ctr'] = (ctr.click / ctr.imps) * 100
    dataframes.append(ctr)

plt.plot(dataframes[0]['ctr'], c='C0',label='user:'+str(advs[0]), marker='*')
plt.plot(dataframes[1]['ctr'], c='C1',label='user:'+str(advs[1]), marker='o')
plt.plot(dataframes[2]['ctr'], c='C2',label='user:'+str(advs[2]), marker='.')
plt.plot(dataframes[3]['ctr'], c='C3',label='user:'+str(advs[3]), marker='>')
plt.plot(dataframes[4]['ctr'], c='C4',label='user:'+str(advs[4]), marker='<')
plt.plot(dataframes[5]['ctr'], c='C5',label='user:'+str(advs[5]), marker='>')
plt.plot(dataframes[6]['ctr'], c='C6',label='user:'+str(advs[6]), marker='s')
plt.plot(dataframes[7]['ctr'], c='C7',label='user:'+str(advs[7]), marker='+')
plt.plot(dataframes[8]['ctr'], c='C8',label='user:'+str(advs[8]), marker='*')
plt.ylabel('CTR')
plt.xlabel('Hours')
plt.xticks(df.hour.unique())
plt.margins(0.05)
plt.legend()
#plt.show()
plt.savefig('CTR_ALL_HOUR.png',dpi=400)



