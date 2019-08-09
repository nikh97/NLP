import os
import re
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
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

import enchant
SPELLING_DICT = enchant.Dict("en_US")

TOTAL_ACCURACY = 0
TOTAL_F_SCORE = 0
TOTAL_AUC = 0
TOTAL_TRIALS = 0

# from nltk.tokenize import sent_tokenize
# from nltk import word_tokenize, pos_tag, ne_chunk

def load_data():
    data = pd.read_csv('yelp_filtered_hotels.csv', encoding = "utf8")
    data = data.drop(columns="date")
    data = data.drop(columns="reviewID")
    data = data.drop(columns="reviewerID")
    data = data.drop(columns="rating")
    data = data.drop(columns="usefulCount")
    data = data.drop(columns="coolCount")
    data = data.drop(columns="funnyCount")
    data = data.drop(columns="hotelID")

    data.rename(columns={'flagged':'real'}, inplace=True)
    data.rename(columns={'reviewContent':'text'}, inplace=True)

    # stop = stopwords.words('english')
    # data['text'] = data['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

    data.loc[data['real'] == 'Y', 'real']  = 0
    data.loc[data['real'] == 'N', 'real']  = 1
    data.loc[data['real'] == 'YR', 'real']  = 0
    data.loc[data['real'] == 'NR', 'real']  = 1

    return data
    
def baseline(data, training):

    review_vectors = []

    print(data.shape)
    current = 0
    for idx, review in data.iterrows():

        if current % 1000 == 0:
            print(current)

        rev = review['text']
        vector = {}

        toks = nltk.word_tokenize(rev)
        pos_tags = nltk.pos_tag(toks)

        for t in toks:

            if training is None:
                if t in vector.keys():
                    vector[t] += 1
                else:
                    vector[t] = 1
            else:
                if t in training:
                    if t in vector.keys():
                        vector[t] += 1
                    else:
                        vector[t] = 1

        if review['real']:
            vector['real'] = 1
        else:
            vector['real'] = 0

        review_vectors.append(vector)

        current += 1

    review_vectors_df = pd.DataFrame.from_dict(review_vectors)
    features = list(review_vectors_df)
    if training is not None:

        for f in training:
            if f not in features:
                review_vectors_df[f] = 0

    review_vectors_df.fillna(0, inplace=True)
    return review_vectors_df


def review_features(review, training_features):
	# training features is list of columns
	vector = {}
	words = review.split(" ")
	tokens = nltk.word_tokenize(review)
	pos = nltk.pos_tag(tokens)
	first_person = 0
	misspelled_words = 0
    
	vector['all_caps'] = 0
	vector['title'] = 0
	vector['number'] = 0
	vector['alphanumeric'] = 0

	for i in range(len(pos)):
		curr_word, curr_tag = pos[i]


		if curr_word.lower() in ['we', 'i', 'me', 'us']:
			first_person = first_person + 1
		if re.match('[a-zA-Z]', curr_tag) and SPELLING_DICT.check(curr_word) is not True:
		  	misspelled_words = misspelled_words + 1

		if curr_tag in vector.keys():
			vector[curr_tag] = vector[curr_tag] + 1
		else:
			vector[curr_tag] = 1

		# check capitalization
		if curr_word.isupper():
			vector['all_caps'] += 1
		if curr_word.istitle():
			vector['title'] += 1
		if curr_word.isdigit():
			vector['number'] += 1
		if re.match('(([a-zA-Z]+[0-9]+([a-zA-Z]+)?)|([0-9]+[a-zA-Z]+))', curr_word):
			vector['alphanumeric'] += 1

	vector['length'] = len(review)
	vector['word_count'] = len(words)
	vector['average_word_length'] = sum(len(x) for x in words) / len(words)
	vector['first_person_freq'] = first_person / float(len(words))
	vector['misspell_freq'] = misspelled_words / float(len(words))

	vector['number_oov'] = 0
	word_set = set()

	for word in tokens:
		if tokens.count(word) == 1:
			vector['number_oov'] += 1
		word_set.add(word)
	vector['unique_word_count'] = len(word_set)

	return vector

def featurize(data, training):

    if training is None:
        training_features = None
    else:
        training_features = list(training)
    review_vectors = []

    print(data.shape)
    current = 0

    for idx, review in data.iterrows():

        if current % 1000 == 0:
             print(current)

        rev = review['text']
        featurized_vector = review_features(rev, training_features)
        
        if review['real']:
            featurized_vector['real'] = 1
        else:
            featurized_vector['real'] = 0


        review_vectors.append(featurized_vector)

        current += 1

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
	x = data.loc[:, data.columns != 'real']
	y_true = data['real']
	y_pred = model.predict(x)
	y_pred_prob = model.predict_proba(x)

	print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
	print("F-score: " + str(f1_score(y_true, y_pred)))
	print("AUC: " + str(roc_auc_score(y_true, y_pred_prob[:,1])))




def main():

    data = load_data()

    i = 0

    while i < 5:
        print ("TRIAL #" + str(i))
        # training = data.sample(frac=0.7)
        # validation = data.loc[set(data.index) - set(training.index)].sample(frac=0.5)
        # test = data.loc[set(data.index) - set(training.index) - set(validation.index)]

        training = data.sample(frac=0.01)
        validation = data.loc[set(data.index) - set(training.index)].sample(frac=0.0025)
        test = data.loc[set(data.index) - set(training.index) - set(validation.index)].sample(frac=0.0025)

        print("Creating baseline..........................")
        print("Creating training baseline")
        training_baseline = baseline(training, None)
        print("Creating validation baseline")
        validation_baseline = baseline(validation, training_baseline)
        print("Creating test baseline")
        test_baseline = baseline(test, training_baseline)

        print("Baseline results...")
        try:        
            model_baseline = make_model(training_baseline)
            evaluate_model(model_baseline, validation_baseline)
            print("Results for Test Model")
            evaluate_model(model_baseline, test_baseline)
            TOTAL_TRIALS += 1
        except ValueError:
            print ("BASELINE ERROR")
        

        print("Adding features...")
        print("Creating featurized training")
        training_featurized = featurize(training, training = None)
        print("Creating featurized validation")
        validation_featurized = featurize(validation, training = training_featurized)
        print("Creating featurized test")
        test_featurized = featurize(test, training = training_featurized)

        print("Featurized results...")
        try:
            model_featurized = make_model(training_featurized)
            evaluate_model(model_featurized, validation_featurized)
            print("Results for Test Model")
            evaluate_model(model_featurized, test_featurized)

        except ValueError:
            print("FEATURIZED ERROR")

        i = i + 1


main()
