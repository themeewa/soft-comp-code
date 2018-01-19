import csv
import math
import random
import copy

def loadCsv(filename, fold):
    rows = csv.reader(open(filename, 'r'))
    filedata = list(rows)
    for i in range(1,len(filedata)):
        if "Iris-setosa" in filedata[i]:
            filedata[i][4] = 0
        elif "Iris-versicolor" in filedata[i]:
            filedata[i][4] =1
        else:
            filedata[i][4] = 2
        filedata[i] = [float(x) for x in filedata[i]]

    # for i in range(1,len(filedata)):
    #     if "Yes" in filedata[i]:
    #         filedata[i][44] = 0
    #     elif "No" in filedata[i]:
    #         filedata[i][44] =1
    #     filedata[i] = [float(x) for x in filedata[i]]

    # spliting dataset using fold 
    fileSize = len(filedata)
    trainingSet = []
    testingSet = []

    tempset = list(filedata)
    # tot = len(tempset)

    for index in range(1, fileSize,1):
        if (index>=fold*10) and (index<(fold+1)*10):
           testingSet.append(tempset[index])
        else:
        	trainingSet.append(tempset[index])
    return [filedata,trainingSet, testingSet]


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stddev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        val = vector.pop()
        separated[val].append(vector)
    return separated


def summarize(dataset):
    summaries = [(mean(attribute), stddev(attribute)) for attribute in zip(*dataset)]
    return summaries

def clean(dataset):
    for i in range(len(dataset)):
        dataset[i].pop(4)
        dataset[i] = dataset[i]
    return dataset

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stddev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stddev,2))))
    val = (1 / (math.sqrt(2*math.pi * stddev))) * exponent
    return (1 / (math.sqrt(2*math.pi * stddev))) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stddev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stddev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def getConfMatrix(testSet, predictions):
	tp,fp,tn,fn = 0,0,0,0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			if testSet[i][-1] == 0:
				print("tp found")
				tp = tp + 1
			elif testSet[i][-1] == 1:
				print("tn found")
				tn = tn + 1
		elif testSet[i][-1] != predictions[i]:
			# print("value not matched")
			if testSet[i][-1] == 1:
				print("fn found")
				fn = fn+1
			elif testSet[i][-1] == 0:
				print("fp found")
				fp = fp + 1
	print("tn:",tn)
	print("tp:",tp)
	print("fn:",fn)
	print("fp:",fp)
	accuracy = (tp+tn)/(tp+tn+fn+fp)
	if (tp+fp) == 0:
		precision = 0
	else:
		precision = tp/(tp+fp)
	if (tp+fn) == 0:
		recall = 0
	else:
		recall = tp/(tp+fn)
	return accuracy, precision, recall

def main():
	accuracy, fprecision, frecall = 0,0,0
	# filename = "SPECTF_test.csv"
	filename = "IRIS.csv"
	totAccuracy = 0
	# running 10 fold
	for x in range(10):
		dataset, trainingSet, testSet = loadCsv(filename,x)
		print('file loaded: {0} with {1} rows'.format(filename, len(dataset)))
		testSet1 = copy.deepcopy(testSet)
		testSet = clean(testSet)
		print('Splitted {0} rows into {1} rows training set and {2} rows testing data set'.format(len(dataset), len(trainingSet), len(testSet)))
		summaries = summarizeByClass(trainingSet)
		predictions = getPredictions(summaries, testSet)
		# accuracy = getAccuracy(testSet1, predictions)
		accuracy, fprecision, frecall = getConfMatrix(testSet1, predictions)
		print('Accuracy: {0} '.format(accuracy))
		totAccuracy = totAccuracy+accuracy
	print((totAccuracy/10))
main()
# if __name__ == '__main__':
#     thersold = 0
#     for i in range(10):
#         accuracy = main()
#         if accuracy >= thersold:
#             thersold = accuracy
#     print('Accuracy: {0}%'.format(accuracy))