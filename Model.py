__author__ = 'root'
import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer

def check(x):
    if str(x).startswith("RT"):
        return False
    return True

def getTimeStamp(x):
    pass

Train=pd.read_csv("/media/distro/New Volume/EduStuff/DataSets/SuperBowlProcessed/Train.csv",sep="|")
Test=pd.read_csv("/media/distro/New Volume/EduStuff/DataSets/SuperBowlProcessed/Test.csv",sep="|")
TrainTime=Train['timestamp']
TestTime=Test['timestamp']
Train=Train.drop(['username','timestamp','retweetcount','lon','lat','country','name','address','type','placeURL'],axis=1)
Test=Test.drop(['username','timestamp','retweetcount','lon','lat','country','name','address','type','placeURL'],axis=1)
Train=Train[Train['tweet'].map(check)]
print Train.shape
# tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word',
#                       ngram_range=(1, 2), stop_words = 'english',max_features=None)

cv=CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 5000)
traindata=np.ravel(Train.values.tolist())[0:100]
testdata=np.ravel(Test.values.tolist())[0:100]
cv.fit(traindata)
Train=cv.transform(traindata)
Test=cv.transform(testdata)
print type(Train)


