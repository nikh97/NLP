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
	review_vector = {}
	words = review.split(" ")
	tokens = nltk.word_tokenize(review)
	pos_tags = nltk.pos_tag(tokens)
	first_person = 0
	misspelled_words = 0
    
	review_vector['all_caps'] = 0
	review_vector['title'] = 0
	review_vector['number'] = 0
	review_vector['alphanumeric'] = 0

	for i in range(len(pos_tags)):
		curr_word, curr_tag = pos_tags[i]


		if curr_word.lower() in ['i', 'we', 'me', 'us']:
			first_person = first_person + 1
		if re.match('[a-zA-Z]', curr_tag) and SPELLING_DICT.check(curr_word) is not True:
		  	misspelled_words = misspelled_words + 1

		if curr_tag in review_vector.keys():
			review_vector[curr_tag] = review_vector[curr_tag] + 1
		else:
			review_vector[curr_tag] = 1

		# check capitalization
		if curr_word.isupper():
			review_vector['all_caps'] = review_vector['all_caps'] + 1
		if curr_word.istitle():
			review_vector['title'] = review_vector['title'] + 1
		if curr_word.isdigit():
			review_vector['number'] = review_vector['number'] + 1
		if re.match('(([a-zA-Z]+[0-9]+([a-zA-Z]+)?)|([0-9]+[a-zA-Z]+))', curr_word):
			review_vector['alphanumeric'] += 1

	review_vector['length'] = len(review)
	review_vector['word_count'] = len(words)
	review_vector['average_word_length'] = sum(len(x) for x in words) / len(words)
	review_vector['first_person_freq'] = first_person / float(len(words))
	review_vector['misspell_freq'] = misspelled_words / float(len(words))

	review_vector['number_oov'] = 0
	word_set = set()

	for word in tokens:
		if tokens.count(word) == 1:
			review_vector['number_oov'] += 1
		word_set.add(word)
	review_vector['unique_word_count'] = len(word_set)
	# review_vector['bigram_word_OOV'] = 0
	# review_vector['bigram_pos_OOV'] = 0

	return review_vector

def featurize(data, training):

    if training is None:
        training_features = None
    else:
        training_features = list(training)
    review_vectors = []

    for idx, review in data.iterrows():
        rev = review['text']
        featurized_vector = review_features(rev, training_features)

        if review['real']:
            featurized_vector['real'] = 1
        else:
            featurized_vector['real'] = 0

        # for features in featurized_vector:
        #     vector[features] = featurized_vector[features]

        review_vectors.append(featurized_vector)

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
    # logreg = linear_model.LogisticRegression()
    # logreg.fit(x, y)

    # clf = svm.SVC()
    # clf.fit(x, y)

    # knn = neighbors.KNeighborsClassifier(15)
    # knn.fit(x, y)

    nbc = naive_bayes.GaussianNB()
    nbc.fit(x, y)

    # return logreg
    # return clf
    # return knn
    return nbc

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

    i = 0

    while i < 10:
        print ("TRIAL #" + str(i))
        training = data.sample(frac=0.7)
        validation = data.loc[set(data.index) - set(training.index)].sample(frac=0.5)
        test = data.loc[set(data.index) - set(training.index) - set(validation.index)]

        print("Creating baseline...")
        training_baseline = baseline(training, None)
        validation_baseline = baseline(validation, training_baseline)
        test_baseline = baseline(test, training_baseline)

        print("Baseline results...")
        try:        
            model_baseline = make_model(training_baseline)
            evaluate_model(model_baseline, validation_baseline)
            print("Results for Test Model")
            evaluate_model(model_baseline, test_baseline)
        except ValueError:
            print ("BASELINE ERROR")
        

        print("Adding features...")
        training_featurized = featurize(training, training = None)
        validation_featurized = featurize(validation, training = training_featurized)
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

# def baseline_unigrams(data):
#     corpus = []
#     for i in range(0, len(data)):  
        
#         # column : "Review", row ith 
#         review = re.sub('[^a-zA-Z]', ' ', data['text'][i])  
        
#         # convert all cases to lower cases 
#         review = review.lower()  
        
#         # split to array(default delimiter is " ") 
#         review = review.split()  
        
#         # creating PorterStemmer object to 
#         # take main stem of each word 
#         ps = PorterStemmer()  
        
#         # loop for stemming each word 
#         # in string array at ith row     
#         review = [ps.stem(word) for word in review]  
                    
#         # rejoin all string array elements 
#         # to create back into a string 
#         review = ' '.join(review)   
        
#         # append each string to create 
#         # array of clean text  
#         corpus.append(review)
#     return corpus   

# def baseline_model(data, corpus):
#     # To extract max 1500 feature. 
#     # "max_features" is attribute to 
#     # experiment with to get better results 
#     cv = CountVectorizer(max_features = 1600)  
    
#     # X contains corpus (dependent variable) 
#     X = cv.fit_transform(corpus).toarray() 

#     # print(cv.get_feature_names())
    
#     # y contains answers if review 
#     # is real or fake 
#     y = data.iloc[:, 1].values  

#     return X, y

main()
