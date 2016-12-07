'''
Created on Oct 05, 2016

@author: Jin
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocalbList (dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

### 1, set-of-words model
def setOfWords2Vec (vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else: print ("the word: %s is not in my Vocabulary!" % word)
	return returnVec

### 2, bag-of-words model
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def trainNB0(trainMatrix, trainCategory):
	numDoc = len(trainMatrix)
	numWords = len(trainMatrix[0])
	P1 = sum(trainCategory) / numDoc
	P0_nu = ones(numWords)
	P0_de = 2
	P1_nu = ones(numWords)
	P1_de = 2
	for i in range(numDoc):
		if trainCategory[i] == 0:
			P0_nu += trainMatrix[i]
			P0_de += sum(trainMatrix[i])
		else:
			P1_nu += trainMatrix[i]
			P1_de += sum(trainMatrix[i])
	Pw_0 = log(P0_nu / P0_de)
	Pw_1 = log(P1_nu / P1_de)
	return Pw_1, Pw_0, P1

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocalbList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
	testEntry = ["love", "my", "dalmation"]
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, "classified as :", classifyNB(thisDoc, p0V, p1V, pAb))
	testEntry = ["stupid", "garbage"]
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, "classified as :", classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()

def textParse(bigString):
	import re
	listOfTokens = re.split("\W+", bigString)
	listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
	return listOfTokens

# test
# test = textParse("Haha, ni shi yi ge hao ren!!!")
# print(test)

def spamTest():
	import codecs; import random
	docList = []; classList = []; fullText = []
	for i in range(1, 26):
		with codecs.open("./email/spam/%d.txt" % i, "r", encoding='utf-8', errors="ignore") as email:
			wordList = textParse(email.read())
			docList.append(wordList)
			fullText.extend(wordList)
			classList.append(1)

		with codecs.open("./email/ham/%d.txt" % i, "r", encoding='utf-8', errors="ignore") as email:
			wordList = textParse(email.read())
			docList.append(wordList)
			fullText.extend(wordList)
			classList.append(0)

	vocabList = createVocalbList(docList)
	trainingSet = list(range(50)); testSet = []

	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
		p1V, p0V, pSpam = trainNB0(array(trainMat), array(trainClasses))

	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		print("prediction:", classifyNB(array(wordVector), p0V, p1V, pSpam), " fact:", classList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1

	print("the error rate is: ", float(errorCount) / len(testSet))

spamTest()




















