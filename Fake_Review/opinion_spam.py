import os
import re
import pandas as pd
# Natural Language Tool Kit 
import nltk
  
# to remove stopword 
from nltk.corpus import stopwords 

from nltk.stem.porter import PorterStemmer 
from sklearn import linear_model, svm, neighbors, naive_bayes
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
# UNCOMMENT THIS
# import enchant

# from nltk.tokenize import sent_tokenize
# from nltk import word_tokenize, pos_tag, ne_chunk

def load_data():
    data = pd.read_csv('deceptive-opinion.csv')
    data = data.drop(columns="hotel")
    data = data.drop(columns="source")
    data = data.drop(columns="polarity")

    data.rename(columns={'deceptive':'real'}, inplace=True)

    stop = stopwords.words('english')

    data['text'] = data['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

    data.loc[data['real'] == 'truthful', 'real']  = 1
    data.loc[data['real'] == 'deceptive', 'real']  = 0

    return data

def baseline(data, training):

    review_vectors = []

    for idx, review in data.iterrows():

        rev = review['text']
        vector = {}

        toks = nltk.word_tokenize(rev)
        pos_tags = nltk.pos_tag(toks)

        for pt in pos_tags:

            word, tag = pt

            if training is None:
                if word in vector.keys():
                    vector[word] += 1
                else:
                    vector[word] = 1
            else:
                columns = list(training.columns.values)
                if word in columns:
                    if word in vector.keys():
                        vector[word] += 1
                    else:
                        vector[word] = 1

        if review['real']:
            vector['real'] = 1
        else:
            vector['real'] = 0

        review_vectors.append(vector)

    review_vectors_df = pd.DataFrame.from_dict(review_vectors)

    if training is not None:

        features = list(review_vectors_df.columns.values)
        columns = list(training.columns.values)


        for f in columns:
            if f not in features:
                review_vectors_df[f] = 0

    review_vectors_df.fillna(0, inplace=True)
    return review_vectors_df


def make_model(data):

    x = data.loc[:, data.columns != 'real']
    y = data['real']
    logreg = linear_model.LogisticRegression()
    logreg.fit(x, y)

    # clf = svm.SVC()
    # clf.fit(x, y)

    # knn = neighbors.KNeighborsClassifier(15)
    # knn.fit(x, y)

    # nbc = naive_bayes.GaussianNB()
    # nbc.fit(x, y)

    return logreg
    # return clf
    # return knn
    # return nbc

def evaluate_model(model, data):
	print("Evaluating model...")
	x = data.loc[:, data.columns != 'real']
	y_true = data['real']
	y_pred = model.predict(x)
	y_pred_prob = model.predict_proba(x)

	print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
	print("F-score: " + str(f1_score(y_true, y_pred)))
	print("AUC: " + str(roc_auc_score(y_true, y_pred_prob[:,1])))


def main():

    data = load_data()

    training = data.sample(frac=0.6)
    validation = data.loc[set(data.index) - set(training.index)].sample(frac=0.5)
    test = data.loc[set(data.index) - set(training.index) - set(validation.index)]

    training_baseline = baseline(training, None)
    validation_baseline = baseline(validation, training_baseline)

    model = make_model(training_baseline)
    evaluate_model(model, validation_baseline)

    # corpus = baseline_unigrams(data)

    # X, y = baseline_model(data, corpus)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 

    # logreg = linear_model.LogisticRegression()
    # logreg.fit(X_train,y_train)

    # model = RandomForestClassifier(n_estimators = 501, 
    #                         criterion = 'entropy') 
                              
    # model.fit(X_train, y_train) 
    # y_pred = model.predict(X_test) 
  
    # cm = confusion_matrix(y_test, y_pred) 
    # print(cm)




main()