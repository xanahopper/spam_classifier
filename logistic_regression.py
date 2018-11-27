import os

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_ROOT = '/Users/xana/Dev/data/aclImdb'
VOCAB_NAME = 'imdb.vocab'
ERROR_NAME = 'imdbEr.txt'
TRAIN_PATH = 'train'
TEST_PATH = 'test'


def get_vocab(path):
    with open(path, 'r') as f:
        vocab = [word.strip() for word in f]
    return vocab


def get_err(path):
    with open(path, 'r') as f:
        err = [int(word.strip()) for word in f]
    return err


def load_data_set(path, label):
    return datasets.load_files(path, label, ['neg', 'pos'], random_state=2)


def tokenized(train_data, test_data, vocab):
    vectorizer = CountVectorizer(vocabulary=vocab)
    x_train = vectorizer.fit_transform(train_data.data)
    x_test = vectorizer.transform(test_data.data)
    return x_train, x_test


def classifier_report(classifier, train_data, test_data, vocab):
    x_train, x_test = tokenized(train_data, test_data, vocab)
    classifier.fit(x_train, train_data.target)
    predict = classifier.predict(x_test)
    print(classification_report(test_data.target, predict, target_names=train_data.target_names))


if __name__ == '__main__':
    vocab = get_vocab(os.path.join(DATA_ROOT, VOCAB_NAME))
    train_data = load_data_set(os.path.join(DATA_ROOT, TRAIN_PATH), 'Train')
    test_data = load_data_set(os.path.join(DATA_ROOT, TEST_PATH), 'Test')

    solvers = ('lbfgs', 'newton-cg', 'sag')
    for s in solvers:
        print(s)
        logistic_regression = LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True,
                                                 intercept_scaling=1, class_weight='balanced', random_state=None,
                                                 solver=s, max_iter=3000, multi_class='ovr', verbose=0,
                                                 warm_start=False, n_jobs=-1)
        classifier_report(logistic_regression, train_data, test_data, vocab)