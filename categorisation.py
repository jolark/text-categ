#!/usr/bin/env python3.2
# -*- coding: utf-8 -*-

'''
Module X9IT090
Fouille de textes et RI

TP 1

J. Lark
'''

import nltk
import tfidf
from nltk.corpus import reuters
from nltk.classify import *
from nltk.corpus import stopwords
from collections import defaultdict
import operator
from progressbar import ProgressBar,Percentage,Bar


# feature vector size
print('building train & test sets...')
N = 10
train_set = []
test_set = []
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(reuters.categories())).start()
k = 0
for cat in reuters.categories():
	k += 1
	pbar.update(k)

	dfs = defaultdict(lambda:0)
	tfs = defaultdict(lambda:defaultdict(lambda:0))
	tfsidfs = defaultdict(lambda:0)
	for file_id in reuters.fileids(cat):
		fileWords = []
		for w in set(reuters.words(file_id)) - set(stopwords.words('english')):
			if w not in fileWords:
				dfs[w] += 1
				fileWords.append(w)
			tfs[file_id][w] += 1
	for file_id in tfs:
		for w in tfs[file_id]:
			tfsidfs[w] = float(tfs[file_id][w])/dfs[w]
		tfidfSorted = dict(sorted(tfsidfs.iteritems(), key=operator.itemgetter(1), reverse=True)[:N])
		if file_id.startswith('train'):
			train_set.append((tfidfSorted,cat))
		else:
			test_set.append((tfidfSorted,cat))
print('done')

print('classifying...')
classifier = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier,test_set))