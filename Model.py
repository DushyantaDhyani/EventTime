__author__ = 'root'

import math
import datetime
import numpy as np
import pandas as pd
import time
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from Constants import DATASET_DATE_FORMAT
from Constants import BASETIME

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

def check(x):
    if str(x).startswith("RT"):
        return False
    return True

def getTimeStamp(x):
    try:
        currenttimestamp=int(  time . mktime(datetime.datetime.strptime(x,DATASET_DATE_FORMAT).timetuple()))
        return (currenttimestamp-basetimestamp)
    except:
        print 'Time entered '+str(x)+' is of incorrect format'
        exit(1)

def getTrainTestData(TrainFile,TestFile):
    Train=pd.read_csv(TrainFile,sep="|")
    Test=pd.read_csv(TestFile,sep="|")
    Train=Train[Train['tweet'].map(check)]
    Train=Train[pd.notnull(Train['timestamp'])]
    Test=Test[pd.notnull(Test['timestamp'])]
    Train=Train[Train['timestamp']!='0']
    Test=Test[Test['timestamp']!='0']
    TrainTime=Train['timestamp']
    TestTime=Test['timestamp']
    Train=Train[['tweet']]
    Test=Test[['tweet']]
    return (Train,Test,TrainTime,TestTime)

basetimestamp=int(time.mktime(datetime.datetime.strptime(BASETIME,DATASET_DATE_FORMAT).timetuple()))

Train,Test,TrainTime,TestTime=getTrainTestData("Data/Trainsmall1.csv","Data/Testsmall.csv")
tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word',
                      ngram_range=(1, 2), stop_words = 'english',max_features=10000)

traindata=np.ravel(Train.values.tolist())
testdata=np.ravel(Test.values.tolist())

tfv.fit(traindata)
Train=tfv.transform(traindata)
Test=tfv.transform(testdata)

TrainTime=TrainTime.map(getTimeStamp)
TestTime=TestTime.map(getTimeStamp)

clf = linear_model.SGDRegressor(random_state=22)
print 'Creating Model'
clf.fit(Train,TrainTime)
joblib.dump(clf,'Data/classifier.pkl')
joblib.dump(tfv,'Data/vectorizer.pkl')
print 'Predicting'
Results=clf.predict(Test)
Total=0
for x,y in zip(Results,TestTime):
    Total+=math.fabs(x-y)

diff= Total*1.0/(len(Results))
# st = datetime.datetime.fromtimestamp(int(diff)).strftime('%Y-%m-%d %H:%M:%S')
print diff/3600