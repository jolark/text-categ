#!/usr/bin/env python3.2
# -*- coding: utf-8 -*-

'''
Module X9IT090
Fouille de textes et RI

TP 1

J. Lark


usage:

python categorisation.py <N> <selection> <classifier>

with:
	N = number of relevant terms by text
	selection = 'freq' or 'chi2' (feature selection method)
	classifier = 'MultinomialNB' or 'GaussianNB' or 'SVM' or 'DecisionTree' or 'KNN'
'''
import sys

import nltk
from nltk.corpus import reuters
from nltk.classify import *
from nltk.corpus import stopwords

from collections import defaultdict

import operator

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

import shelve

def main():
	if len(sys.argv) != 4 or not sys.argv[2] in ['freq','chi2']  or not sys.argv[3] in ['MultinomialNB','GaussianNB','SVM','DecisionTree','KNN']:
		print('usage:\n\
python categorisation.py <N> <selection> <classifier>\n\n\
with:\n\
N = number of relevant terms by text\n\
selection = \'freq\' or \'chi2\' (feature selection method)\n\
classifier = \'MultinomialNB\' or \'GaussianNB\' or \'SVM\' or \'DecisionTree\' or \'KNN\'')
		exit()

	train_set = []
	test_set = []

	N = sys.argv[1]
	method = sys.argv[2]
	s = shelve.open('featuresDictonaries')
	if ('train_'+str(N)) in s:
		train_set = s['train_'+str(N)]
	if ('test_'+str(N)) in s:
		test_set = s['test_'+str(N)]
	s.close()

	if not (train_set and test_set):
		print('building freq features dictionaries...')
		for cat in reuters.categories():
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
				tfidfSorted = dict(sorted(tfsidfs.iteritems(), key=operator.itemgetter(1), reverse=True)[:int(N)])
				if file_id.startswith('train'):
					train_set.append((tfidfSorted,cat))
				else:
					test_set.append((tfidfSorted,cat))
		print('done')
		
		print('saving to featuresDictonaries...')
		s = shelve.open('featuresDictonaries')		
		s['train_'+str(N)] = train_set
		s['test_'+str(N)] = test_set
		s.close()
		print('done')



	print('classifying...')

	pipeline = Pipeline([('chi2', SelectKBest(chi2, k=290)),('svm', svm.LinearSVC())])
	classifier = SklearnClassifier(pipeline)

	# chi2
	if sys.argv[2] == 'chi2':
		if sys.argv[3] == 'KNN':
			pipeline = Pipeline([('chi2', SelectKBest(chi2, k=290)),('svm', KNeighborsClassifier(n_neighbors=5))])
			classifier = SklearnClassifier(pipeline)
		elif sys.argv[3] == 'MultinomialNB':
			pipeline = Pipeline([('chi2', SelectKBest(chi2, k=290)),('svm', MultinomialNB())])
			classifier = SklearnClassifier(pipeline)
		elif sys.argv[3] == 'GaussianNB':
			pipeline = Pipeline([('chi2', SelectKBest(chi2, k=290)),('svm', GaussianNB())])
			classifier = SklearnClassifier(pipeline,sparse=False)
		elif sys.argv[3] == 'DecisionTree':
			pipeline = Pipeline([('chi2', SelectKBest(chi2, k=290)),('svm', tree.DecisionTreeClassifier())])
			classifier = SklearnClassifier(pipeline,sparse=False)
	else:
		if sys.argv[3] == 'KNN':
			classifier = SklearnClassifier(KNeighborsClassifier(n_neighbors=5))
		elif sys.argv[3] == 'MultinomialNB':
			classifier = SklearnClassifier(MultinomialNB())
		elif sys.argv[3] == 'GaussianNB':
			classifier = SklearnClassifier(GaussianNB(),sparse=False)
		elif sys.argv[3] == 'DecisionTree':
			classifier = SklearnClassifier(tree.DecisionTreeClassifier(),sparse=False)
		elif sys.argv[3] == 'SVM':
			classifier = SklearnClassifier(svm.LinearSVC())

	classifier.train(train_set)

	test_skl = []
	t_test_skl = []
	for d in test_set:
		test_skl.append(d[0])
		t_test_skl.append(d[1])

	p = classifier.batch_classify(test_skl)

	print classification_report(t_test_skl, p, labels=list(set(t_test_skl)),target_names=reuters.categories())

if __name__ == '__main__':
	main()