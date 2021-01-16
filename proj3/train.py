import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

if __name__ == '__main__':

    train_label, train_data, test_label, test_data = [], [], [], []
    with open("train.json") as file:
        for e in json.load(file):
            train_data.append(e['data'])
            train_label.append(e['label'])

    with open("test.json") as file:
        for e in json.load(file):
            test_data.append(e['review'])
            test_label.append(e['sentiment'])

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    print("Start vectorization!")
    vectorizer = TfidfVectorizer(lowercase=False,max_df=0.71, sublinear_tf=True, ngram_range=(1, 2)).fit(np.hstack(train_data))
    train_data = vectorizer.fit_transform(train_data)
    test_data = vectorizer.transform(test_data)
    print("Start training!")

    for e in range(11):
        c = 0.26 + e * 0.01

        for penalty in ['l2']:

            svc = LinearSVC(C=c, penalty=penalty)
            svc.fit(train_data, train_label)

            predictions = svc.predict(test_data)
            cnt = 0
            for i in range(len(predictions)):
                if predictions[i] == test_label[i]:
                    cnt += 1

            print(c, penalty, " : ", cnt / len(predictions))
