
import string
import math
import sys
import nltk
import operator
from stop_list import closed_class_stop_words as stop_list


abstract_dict = {}
queries_dict = {}
idf_dict = {}

class abstract:
	def __init__(self, num):
		self.num = num
		self.words = []
		self.TFIDF = {}

class query:
	def __init__(self, num):
		self.num = num
		self.words = []
		self.TFIDF = {}	
		self.cosineSim = {}	

def get_abstracts(filename):

	with open(filename, 'r') as f:

		words = []
		current_abs = None
		count = 1

		for l1 in f:
			l1 = l1.strip().split()

			if ('.W' in l1):

				l2 = f.readline().strip()
				while ('.I' not in l2):

					tokens = nltk.word_tokenize(l2)
					for t in tokens:
						words.append(t)
					l2 = f.readline().strip()
					if (l2 == ''):
						break

				l2 = l2.split()
				current_abs = abstract(count)

				for word in words[:]:

					if word in string.punctuation:
						words.remove(word)
					if word.isdigit():
						words.remove(word)
					if word in stop_list:
						words.remove(word)

				current_abs.words = words
				abstract_dict[count] = current_abs
				count += 1

			words = []

def get_queries(filename):

	with open(filename, 'r') as f:

		words = []
		current_q = None
		count = 1

		for l1 in f:
			l1 = l1.strip().split()

			if ('.W' in l1):

				l2 = f.readline().strip()
				while ('.I' not in l2):

					tokens = nltk.word_tokenize(l2)
					for t in tokens:
						words.append(t)
					l2 = f.readline().strip()
					if (l2 == ''):
						break

				l2 = l2.split()
				current_q = query(count)

				for word in words[:]:

					if word in string.punctuation:
						words.remove(word)
					if word.isdigit():
						words.remove(word)
					if word in stop_list:
						words.remove(word)

				current_q.words = words
				queries_dict[count] = current_q
				count += 1

			words = []

def calc_idf(abstract_dict):

	num_docs = len(abstract_dict)

	for abstract in abstract_dict:
		for word in abstract_dict[abstract].words:
			if word not in idf_dict:
				idf_dict[word] = 0

	for word in idf_dict:

		doc_freq = 0

		for abstract in abstract_dict:

			if word in abstract_dict[abstract].words:
				doc_freq += 1

		idf_dict[word] = math.log(num_docs/doc_freq)

def calc_tfidf(dictionary):

	for i in dictionary:

		obj = dictionary[i]

		for word in obj.words:

			if word not in obj.TFIDF:
				obj.TFIDF[word] = 1
			else:
				obj.TFIDF[word] += 1
			obj.TFIDF[word] = (obj.TFIDF[word]/len(obj.words))
				
		for word in obj.TFIDF:

			IDF = None
			try:	
				IDF = idf_dict[word]
			except:
				IDF = 0

			obj.TFIDF[word] *= IDF

def cosine(q_vector, a_vector):

	numerator = 0
	a_2_sum = 0
	q_2_sum = 0

	for i in range(len(q_vector)):

		numerator += q_vector[i]*a_vector[i]
		a_2_sum += a_vector[i]*a_vector[i]
		q_2_sum += q_vector[i]*q_vector[i]

	if (a_2_sum == 0 or q_2_sum == 0):
		return 0
	else:
		return numerator/(math.sqrt(a_2_sum*q_2_sum))

def cos_sim(query, abstract):

	a_vector = []
	q_vector = []
	cos_sim = 0

	for word in query.TFIDF:

		q_vector.append(query.TFIDF[word])
		try:
			a_vector.append(abstract.TFIDF[word])
		except:
			a_vector.append(0)

	cos_sim = cosine(q_vector, a_vector)

	query.cosineSim[abstract.num] = cos_sim

def out(q, file):

	num = q.num
	cosineSim = q.cosineSim

	sorted_cos_sim = sorted(cosineSim.items(), key = operator.itemgetter(1), reverse=True)

	for word in sorted_cos_sim:

		file.write(str(num) + ' ' + str(word[0]) + ' ' + str(word[1]) + '\n')

	
	

def main():

	if(len(sys.argv) != 2):
		print("Incorrect usage: must include output files in arguments.")
		exit(0)

	input1 = 'cran.all.1400'
	input2 = 'cran.qry'

	output = sys.argv[1]

	get_abstracts(input1)
	get_queries(input2)

	calc_idf(abstract_dict)
	calc_tfidf(abstract_dict)
	calc_tfidf(queries_dict)

	for query in queries_dict:
		for abstract in abstract_dict:
			cos_sim(queries_dict[query], abstract_dict[abstract])

	with open(output, 'w') as f:
		for query in queries_dict:
			out(queries_dict[query], f)
	

if __name__ == "__main__":
	main()