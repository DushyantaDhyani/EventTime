__author__ = 'root'
import pandas as pd


def checkReTweet(x):
    global counterpos
    global counterneg
    try:
        if x.strip().startswith('RT'):
           counterpos+=1
           return False
        else:
            counterneg+=1
            return True
    except AttributeError:
        print "Some issues with "+str(x)

counterpos=0
counterneg=0
Train=pd.read_csv("Data/Train.csv",sep="|")
Test=pd.read_csv("Data/Test.csv",sep="|")
Train=Train.dropna()
Test=Test.dropna()
Train=Train[Train['tweet'].map(checkReTweet)]
Test=Test[Test['tweet'].map(checkReTweet)]
Train.to_csv("Data/NewTrain.csv",sep="|")
Test.to_csv("Data/NewTest.csv",sep="|")