import bayes2

posts, classes = bayes2.loadDataSet()

vocal_list = bayes2.createVocalbList(posts)

wordVec = bayes2.setOfWords2Vec(vocal_list ,posts[0])

# print(wordVec)

# train_matrix = []

# for i in range(len(posts)):
# 	wordVec = bayes2.setOfWords2Vec(vocal_list, posts[i])
# 	train_matrix.append(wordVec)

# print(bayes2.trainNB0(train_matrix, classes))

### a data cleaning trick
mySent = "This book is the best book on Python or M.L. I have ever laid eyes upon."
# print(mySent.split())

# import codecs
# with codecs.open("./email/ham/6.txt", "r",encoding='utf-8', errors='ignore') as email:
# 	print(email.read())

# for i in list(range(10)):
# 	int(random.uniform(0, len([2, 3])))

import random
print(random.uniform(1, 10))

test = [1:3]
del(test[1])
print(test)