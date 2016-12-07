import bayes

posts, classes = bayes.loadDataSet()

vocal_list = bayes.createVocabList(posts)

# wordVec = bayes.setOfWords2Vec(vocal_list ,posts[0])

train_matrix = []

for i in range(len(posts)):
	wordVec = bayes.setOfWords2Vec(vocal_list, posts[i])
	train_matrix.append(wordVec)

print(bayes.trainNB0(train_matrix, classes))