import math
import os
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

DATA_ROOT = '/Users/xana/Dev/data/'
HOT_DATA = 'ivy_l4_hot_entry'
NORMAL_DATA = 'ivy_l4_not_hot_entry'


def load_data_set(data_name, target):
    with open(os.path.join(DATA_ROOT, data_name)) as f:
        for i, line in enumerate(f):
            pass
    features = np.zeros((i + 1, 10))
    with open(os.path.join(DATA_ROOT, data_name)) as f:
        for i, line in enumerate(f):
            tokens = line.strip('\n').split(' ')
            tokens[4] = int(tokens[4] == 'True')
            tokens[6] = int(tokens[6] == 'M')
            tokens[9] = int(tokens[9] == 'True')
            features[i] = tokens
    return features, [target] * features.shape[0]


def normalize_data_set(data):
    for i in range(0, 10):
        data[:, i] = preprocessing.minmax_scale(data[:, i])
    return data


def report_classifier(classifier, data, target, split):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=split)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    hot_data, hot_target = load_data_set(HOT_DATA, 1)
    normal_data, normal_target = load_data_set(NORMAL_DATA, 0)
    data = normalize_data_set(np.concatenate((hot_data, normal_data)))
    target = hot_target + normal_target
    report_classifier(LinearSVC(), data, target, 0.2)
    report_classifier(SVC(), data, target, 0.2)
    report_classifier(SVC(kernel='poly'), data, target, 0.2)
    report_classifier(SVC(kernel='sigmoid'), data, target, 0.2)