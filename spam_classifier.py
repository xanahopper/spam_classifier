import os
import email
import re
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

DATA_ROOT = '/Users/xana/Dev/data/20_newsgroups/alt.atheism/'
DICT_SIZE = 3000
SPLIT_REG = re.compile("[ .,\"';()~!?|#\[\]<>+\-`@_*$:=]")

def get_dir_list(dir):
    return [os.path.join(dir, item) for item in os.listdir(dir)]


def load_dictionary(train_dir):
    mails = get_dir_list(train_dir)
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

    return [key for key, value in dictionary.most_common(DICT_SIZE)]


def extract_features(mail_dir, dictionary):
    mails = get_dir_list(mail_dir)
    print("extracting features: ", len(mails))
    feature_matrx = np.zeros((len(mails), DICT_SIZE))
    for i, path in enumerate(mails):
        with open(path, 'r', encoding="utf8", errors="ignore") as m:
            mail = email.message_from_file(m)
            for line in mail.get_payload().split('\n'):
                words = [word.strip(" .,\"';()~!?|#").lower() for word in re.split(SPLIT_REG, line.strip()) if len(word) > 0]
                for word in words:
                    if word in dictionary:
                        feature_matrx[i, dictionary.index(word)] += 1
        if i % 100 == 99:
            print(i + 1, " extracted")
    return feature_matrx

def classify_kmeans(features, k):
    estimator = KMeans(k)
    estimator.fit(features)
    print(estimator.labels_)
    return estimator.labels_


def classify_knn(features, k):
    nbrs = NearestNeighbors(k).fit(features)



if __name__ == '__main__':
    dictionary = load_dictionary(DATA_ROOT)
    features = extract_features(DATA_ROOT, dictionary)
    classify_kmeans(features, 10)
    a = 1