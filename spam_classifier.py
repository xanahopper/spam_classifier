import os
import email
import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sw

DATA_ROOT = '/Users/xana/Dev/data/20_newsgroups/'
DICT_SIZE = 3000
SPLIT_REG = re.compile("[ .,\"';()~!?|#\[\]<>+\-`@_*$:=]")


def get_dir_list(dir):
    return [os.path.join(dir, item) for item in os.listdir(dir)]


def load_dictionary(mails):
    all_words = []
    for path in mails:
        with open(path, 'r', encoding="utf8", errors="ignore") as m:
            mail = email.message_from_file(m)
            for line in mail.get_payload().split('\n'):
                words = re.split(SPLIT_REG, line.strip())
                all_words += [word.strip(" .,\"';()~!?|#").lower() for word in words if len(word) > 0]

    dictionary = Counter(all_words)
    keys = list(dictionary.keys())
    for item in keys:
        if not item.isalpha():
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        elif item in sw:
            del dictionary[item]

    return [key for key, value in dictionary.most_common(DICT_SIZE)]


def extract_features(name, mails, dictionary):
    print("extracting features: ", len(mails))
    feature_matrx = np.zeros((len(mails), DICT_SIZE))
    for i, path in enumerate(mails):
        with open(path, 'r', encoding="utf8", errors="ignore") as m:
            mail = email.message_from_file(m)
            for line in mail.get_payload().split('\n'):
                words = [word.strip(" .,\"';()~!?|#").lower() for word in re.split(SPLIT_REG, line.strip())
                         if len(word) > 0]
                for word in words:
                    if word in dictionary:
                        feature_matrx[i, dictionary.index(word)] += 1
        if i % 100 == 99:
            print(i + 1, " extracted", end='\r')
    print(d, 'extracted')
    return feature_matrx


def classify_kmeans(features, k):
    return KMeans(k).fit(features).labels_


def classify_knn(features, k):
    return NearestNeighbors(k).fit(features)


def classify_nb(features, y):
    mnb = MultinomialNB()
    mnb.fit(features, y)
    return mnb


if __name__ == '__main__':
    dirs = os.listdir(DATA_ROOT)
    target = []
    features = np.zeros((0, DICT_SIZE))
    target_names = []
    for i, d in enumerate(dirs):
        mails = get_dir_list(os.path.join(DATA_ROOT, d))
        dictionary = load_dictionary(mails)
        print('start extract: ', d, i + 1, "/", len(dirs))
        f = extract_features(d, mails, dictionary)
        features = np.concatenate((features, f))
        target += [i] * len(f)
        target_names += [d]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25)
    mnb = classify_nb(X_train, y_train)
    y_predict = mnb.predict(X_test)
    print(classification_report(y_test, y_predict, target_names=target_names))

