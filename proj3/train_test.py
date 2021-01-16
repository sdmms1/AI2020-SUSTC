import argparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train_file', type=str, default='train.json')
    parser.add_argument('-i', '--test_file', type=str, default='testdataexample')
    args = parser.parse_args()
    # print(args.train_file, args.test_file)

    train_label, train_data, test_data = [], [], []
    with open(args.train_file) as file:
        for e in json.load(file):
            train_data.append(e['data'])
            train_label.append(e['label'])

    with open(args.test_file) as file:
        for e in json.load(file):
            test_data.append(e)

    with open('./output.txt', 'w') as file:
        train_data = np.array(train_data)
        test_data = np.array(test_data)

        # print("Start vectorization!")
        vectorizer = TfidfVectorizer(lowercase=False, max_df=0.7, sublinear_tf=True, ngram_range=(1, 2))
        train_data = vectorizer.fit_transform(train_data)
        test_data = vectorizer.transform(test_data)

        # print("Start training!")
        svc = LinearSVC(C=6, class_weight={0: 0.4, 1: 0.6}, fit_intercept=False)
        svc.fit(train_data, train_label)

        predictions = svc.predict(test_data)
        for i in range(len(predictions)):
            print(predictions[i])
            file.write('%d\n' % predictions[i])
