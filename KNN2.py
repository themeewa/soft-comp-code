
import csv
import random
import math
import operator

#loading dataset from csv file
def loadCsv(fname):
    datafile=csv.reader(open(fname,"r"))
    next(datafile)
    ds=list(datafile)
    for i in range(len(ds)):
        for j in range(len(ds[i])-1):
            ds[i][j]=float(ds[i][j])
    return ds            


#preparing training and test dataset
def splitSet(ds,i):
    print("fold: ",i+1)
    ts=int(len(ds)*0.1)
    tSet=[]
    copy=list(ds)
    while(len(tSet)<ts):
        tSet.append(copy.pop(i*10))
    print(tSet)
    return tSet,copy


#computing distance between given x and y
def computeDistance(x,y):
    z=0
    for i in range(len(x)-1):
        z=z+pow(x[i]-y[i],2)
    z=math.sqrt(z)
    return z    


#core knn function
def KNN(test,trainSet,k):
    distance=[]
    for i in range(len(trainSet)):
            tempdist = []
            tempdist.append(computeDistance(test,trainSet[i]))
            tempdist.append(i)
            distance.append(tempdist)
#     print(distance)
    distance.sort(key=operator.itemgetter(0))
    
#     print (distance)
    a=0 
    b=0
    for i in range(k):
        if(trainSet[distance[k][1]][44] == "Yes"):
            a = a + 1
        else:
            b = b + 1
    if (a>b):
        return "Yes"
    else:
        return "No"

#calculating accuracy of single fold
def getAccuracy(trainSet,testSet):
    predictions =[]
    tp,fp,tn,fn = 0,0,0,0
    correct = 0
    for i in range(len(testSet)):
        predictions.append(KNN(testSet[i],trainSet,3))
    for i in range(len(testSet)):
        if (testSet[i][44] == predictions[i]):
            correct = correct + 1
            if (predictions[i]=="Yes"):
                tp = tp+1
            if (predictions[i]=="No"):
                tn = tn+1
        else:
            if (predictions[i]=="Yes"):
                fp = fp+1
            if (predictions[i]=="No"):
                fn = fn+1
    print("tn:",tn)
    print("tp:",tp)
    print("fn:",fn)
    print("fp:",fp)
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    print("accuracy: ",accuracy)
    if (tp+fp) == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    if (tp+fn) == 0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    return accuracy, precision, recall


#main function
def main():
    filename='SPECTF_test.csv'
    dataset=loadCsv(filename)
    totacc,totprec,totrec,accuracy, precision, recall = 0,0,0,0,0,0
    for i in range (10):
        testSet,trainSet=splitSet(dataset,i)
        accuracy, precision, recall = getAccuracy(trainSet,testSet)
        totacc = totacc+accuracy
        totprec = totprec+precision
        totrec = totrec+recall
    print("==========")
    print("average accuracy: ",totacc/10)
    print("average recall: ",totrec/10)
    print("average precision: ",totprec/10)
    
    


#executing main function
main()

