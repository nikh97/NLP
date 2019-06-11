import sys

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."]

def proper_noun(current, pos):

	for char in current:
		if (char.isupper() and pos in ['NNP', 'NNPS']):
			return 'PROPER_NOUN\t'

	return ''

def proper_noun_group(pos, p_pos_1, n_pos_1):

	if(p_pos_1 in ['NNP', 'NNPS'] or n_pos_1 in ['NNP', 'NNPS']):
		if(pos in ['NNP', 'NNPS']):
			return 'PROPER_NOUN_GROUP\t'

	return ''

def noun_group(pos, n_pos_1):

	if(pos in ['JJ', 'JJS', 'JJR', 'DT', 'CD']):
		if(n_pos_1 in ['JJ', 'JJS', 'JJR', 'NN', 'NNS', 'NNP', 'NNPS', 'CD']):
			return 'NOUN_GROUP\t'

	return ''

def money(word, previous_1):

	if(word == '$' or previous_1 == '$'):
		return 'MONEY\t'

	return ''

def abbreviation(word):

	if('.' in word):
		for char in word:
			if(char.isupper()):

				return 'ABBREVIATION\t'

	return ''

def loc(p_pos_1, pos):

	if(p_pos_1 == 'IN' and pos in ['NNP', 'NNPS']):
		return 'LOC\t'

	return ''

def date(word, previous_1, next_1):

	if (word in months):
		return 'DATE\t'
	if (previous_1 in months or next_1 in months):
		if (word == 'CD'):
			return 'DATE\t'
	return ''

def get_features(word, previous_1, previous_2, next_1, next_2, training):

	pos = word[1]
	bio = None
	if (training):
		bio = word[2]

	feature = word[0] + '\t' + 'POS=' + pos + '\t'

	if (previous_1 != 'N/A'):
		p_pos_1 = previous_1[1]
		feature += 'previous_word=' + previous_1[0] + '\t' + 'previous_1_POS=' + p_pos_1 + '\t'
	else:
		p_pos_1 = ' '

	if (previous_2 != 'N/A'):
		p_pos_2 = previous_2[1]
		feature += 'previous_2_word='+ previous_2[0] + '\t' + 'previous_2_POS=' + p_pos_2 + '\t'
	else:
		p_pos_2 = ' '

	if (next_1 != 'N/A'):
		n_pos_1 = next_1[1]
		feature += 'next_word='+ next_1[0] + '\t' + 'next_1_POS=' + n_pos_1 + '\t'
	else:
		n_pos_1 = ' '

	if(next_2 != 'N/A'):
		n_pos_2 = next_2[1]
		feature += 'next_2_word='+ next_2[0] + '\t' + 'next_2_POS=' + n_pos_2 + '\t'
	else:
		n_pos_2 = ' '

	feature += proper_noun(word[0], pos)

	if (previous_2 == ' ' and next_2 == ' ' and previous_1 != ' ' and next_1 != ' '):
		feature += proper_noun_group(pos, p_pos_1, n_pos_1)
		feature += noun_group(pos, n_pos_1)
		feature += money(word[0], previous_1[0])
		feature += loc(p_pos_1, pos)
		feature += date(word[0], previous_1[0], next_1[0])

	elif (previous_2 == ' ' or previous_1 == ' '):
		feature += proper_noun_group(pos, '', n_pos_1)

		try:
			feature += noun_group(pos, n_pos_1)
		except:
			feature += ''

		try:
			feature += money(word[0], previous_1[0])
		except:
			feature += money(word[0], '')

		try:
			feature += date(word[0], previous_1[0], next_1[0])
		except:
			feature += date(word[0], '', next_1[0])

	elif ((next_2 == ' ' or next_1 == ' ')):
		feature += proper_noun_group(pos, p_pos_1, '')

		try:
			feature += noun_group(pos, n_pos_1)
		except:
			feature += ''

		try:
			feature += money(word[0], previous_1[0])
		except:
			feature += money(word[0], '')

		try:
			feature += date(word[0], previous_1[0], next_1[0])
		except:
			feature += date(word[0], previous_1[0], '')

	else:
		feature += proper_noun_group(pos, p_pos_1, n_pos_1)
		feature += noun_group(pos, n_pos_1)
		feature += money(word[0], previous_1[0])
		feature += loc(p_pos_1, pos)
		feature += date(word[0], previous_1[0], next_1[0])

	feature += abbreviation(word[0])

	if(bio != None):
		feature += bio

	return feature

def create_output(input, output, training):

	line_array = []

	with open(input, 'r') as f:

		for line in f:

			if line == '\n':

				line_array.append('\n')
			else:
				line_array.append((line.strip()).split())



	with open(output, 'w') as f:

		for i in range(len(line_array)):

			word = line_array[i]

			if word == '\n':
				f.write('\n')
				continue

			try:
				if(line_array[i-1] == '\n'):
					previous_1 = ["sentence_break", "sentence_break", '']
				else:
					previous_1 = line_array[i-1]
			except:
				previous_1 = 'N/A'

			try:
				if(line_array[i-2] == '\n' and (i-2 >= 0)):
					previous_2 = ["sentence_break", "sentence_break", '']
				elif (i-2 < 0):
					previous_2 = 'N/A'
				else:
					previous_2 = line_array[i-2]
			except:
				previous_2 = 'N/A'

			try:
				if(line_array[i+1] == '\n'):
					next_1 = ["sentence_break", "sentence_break", '']
				else:
					next_1 = line_array[i+1]
			except:
				next_1 = 'N/A'

			try:
				if(line_array[i+2] == '\n'):
					next_2 = ["sentence_break", "sentence_break", '']
				else:
					next_2 = line_array[i+2]
			except:
				next_2 = 'N/A'

			word_features = get_features(word, previous_1, previous_2, next_1, next_2, training)
			f.write(word_features + '\n')

def main():

	if (len(sys.argv) != 5):
		print("Incorrect usage: must include 2 input files and 2 output files in arguments.")
		print("Usage: training input | training output | feature input | feature output")
		exit(0)

	create_output(sys.argv[1], sys.argv[2], 1)
	create_output(sys.argv[3], sys.argv[4], 0)

if __name__ == "__main__":
	main()
