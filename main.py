from collections import Counter
from datasets import load_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import nltk

from nltk import word_tokenize
from nltk.util import bigrams, trigrams

import re as re

train_data = load_dataset("derek-thomas/ScienceQA", split= 'train')
train_data = train_data.remove_columns(['image','choices','hint','answer','task','subject','topic','category','skill'])

dev_data = load_dataset("derek-thomas/ScienceQA", split= 'validation')
dev_data = dev_data.remove_columns(['image','choices','hint','answer','task','subject','topic','category','skill'])

test_data = load_dataset("derek-thomas/ScienceQA", split= 'test')
test_data = test_data.remove_columns(['image','choices','hint','answer','task','subject','topic','category','skill'])

training_unigrams = []
training_bigrams = []
training_trigrams = []
training_labels = []

for instance in train_data:
    tokens = word_tokenize(instance['lecture'])
    tokens += word_tokenize(instance['question'])
    bigram_list = list(bigrams(tokens))
    trigram_list = list(trigrams(tokens))
    unigram_feature_dict = {x: float(0) for x in tokens}
    bigram_feature_dict = {x: float(0) for x in bigram_list}
    trigram_feature_dict = {x: float(0) for x in trigram_list}
    training_labels.append(instance['grade'])

    for tok in tokens:
        unigram_feature_dict[tok] += 1.0
    training_unigrams.append(unigram_feature_dict)
    for bigram in bigram_list:
        bigram_feature_dict[bigram] += 1.0
    training_bigrams.append(bigram_feature_dict)
    for trigram in trigram_list:
        trigram_feature_dict[trigram] += 1.0
    training_trigrams.append(trigram_feature_dict)

dev_unigrams = []
dev_bigrams = []
dev_trigrams = []
dev_labels = []

for instance in dev_data:
    tokens = word_tokenize(instance['lecture'])
    tokens += word_tokenize(instance['question'])
    bigram_list = list(bigrams(tokens))
    trigram_list = list(trigrams(tokens))
    unigram_feature_dict = {x: float(0) for x in tokens}
    bigram_feature_dict = {x: float(0) for x in bigram_list}
    trigram_feature_dict = {x: float(0) for x in trigram_list}
    dev_labels.append(instance['grade'])

    for tok in tokens:
        unigram_feature_dict[tok] += 1.0
    dev_unigrams.append(unigram_feature_dict)
    for bigram in bigram_list:
        bigram_feature_dict[bigram] += 1.0
    dev_bigrams.append(bigram_feature_dict)
    for trigram in trigram_list:
        trigram_feature_dict[trigram] += 1.0
    dev_trigrams.append(trigram_feature_dict)

test_unigrams = []
test_bigrams = []
test_trigrams = []
test_labels = []

for instance in test_data:
    tokens = word_tokenize(instance['lecture'])
    tokens += word_tokenize(instance['question'])
    bigram_list = list(bigrams(tokens))
    trigram_list = list(trigrams(tokens))
    unigram_feature_dict = {x: float(0) for x in tokens}
    bigram_feature_dict = {x: float(0) for x in bigram_list}
    trigram_feature_dict = {x: float(0) for x in trigram_list}
    test_labels.append(instance['grade'])

    for tok in tokens:
        unigram_feature_dict[tok] += 1.0
    test_unigrams.append(unigram_feature_dict)
    for bigram in bigram_list:
        bigram_feature_dict[bigram] += 1.0
    test_bigrams.append(bigram_feature_dict)
    for trigram in trigram_list:
        trigram_feature_dict[trigram] += 1.0
    test_trigrams.append(trigram_feature_dict)

vectorizer = DictVectorizer()
classifier = MultinomialNB()
mnb_alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

X_train_vector = vectorizer.fit_transform(training_unigrams)
y_train = training_labels
classifier.fit(X_train_vector,y_train)

classifier.score(X_train_vector, training_labels)

X_dev_vector = vectorizer.transform(dev_unigrams)
y_dev = dev_labels
score_dict = {}
for i in mnb_alpha_values:
    classifier = MultinomialNB(alpha = i)
    classifier.fit(X_dev_vector, y_dev)
    score_dict[i] = classifier.score(X_dev_vector,y_dev)

for i in score_dict:
    print(i, score_dict[i])

classifier = MultinomialNB(alpha = 0.1)
X_test_vector = vectorizer.transform(test_unigrams)
y_test = test_labels
classifier.fit(X_test_vector, y_test)
classifier.score(X_test_vector,y_test)

X_train_vector = vectorizer.fit_transform(training_bigrams)
y_train = training_labels
classifier.fit(X_train_vector,y_train)

classifier.score(X_train_vector, training_labels)

X_dev_vector = vectorizer.transform(dev_bigrams)
y_dev = dev_labels
score_dict = {}
for i in mnb_alpha_values:
    classifier = MultinomialNB(alpha = i)
    classifier.fit(X_dev_vector, y_dev)
    score_dict[i] = classifier.score(X_dev_vector,y_dev)

for i in score_dict:
    print(i, score_dict[i])

X_test_vector = vectorizer.transform(test_bigrams)
y_test = test_labels

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_test_vector,y_test)
classifier.score(X_test_vector,y_test)

X_train_vector = vectorizer.fit_transform(training_trigrams)
y_train = training_labels
classifier.fit(X_train_vector,y_train)

classifier.score(X_train_vector, training_labels)

X_dev_vector = vectorizer.transform(dev_trigrams)
y_dev = dev_labels
score_dict = {}
for i in mnb_alpha_values:
    classifier = MultinomialNB(alpha = i)
    classifier.fit(X_dev_vector, y_dev)
    score_dict[i] = classifier.score(X_dev_vector,y_dev)


for i in score_dict:
    print(i, score_dict[i])

X_test_vector = vectorizer.transform(test_trigrams)
y_test = test_labels

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_test_vector,y_test)
classifier.score(X_test_vector,y_test)

