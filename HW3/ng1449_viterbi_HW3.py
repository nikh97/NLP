import sys
import random

def training(filename, likelihood, transition, words_in_training_corpus):

    with open(filename, 'r') as f:

        previous_state = 'begin_sent'

        for line in f:

            line = line.strip()

            if (line == ''):
                transition[previous_state] = transition.get(previous_state, {})
                transition[previous_state]['end_sent'] = transition[previous_state].get('end_sent', 0)
                transition[previous_state]['end_sent'] += 1
                previous_state = 'begin_sent'

            else:
                data = line.split('\t')
                transition[previous_state] = transition.get(previous_state, {})
                transition[previous_state][data[1]] = transition[previous_state].get(data[1], 0)
                transition[previous_state][data[1]] += 1

                likelihood[data[1]] = likelihood.get(data[1], {})
                likelihood[data[1]][data[0]] = likelihood[data[1]].get(data[0], 0)
                likelihood[data[1]][data[0]] += 1

                previous_state = data[1]

                if(data[0] not in words_in_training_corpus):
                    words_in_training_corpus.append(data[0])
          
        transition[previous_state] = transition.get(previous_state, {})
        transition[previous_state]['end_sent'] = transition[previous_state].get('end_sent', 0)
        transition[previous_state]['end_sent'] += 1
    

    for k1 in likelihood:

        dic = likelihood[k1]

        total = sum(dic.values())
        for k2 in dic:

            dic[k2] /= total

    for k1 in transition:

        dic = transition[k1]

        total = sum(dic.values())
        for k2 in dic:

            dic[k2] /= total


def OOV(word, previous_state):

    current_state = random.choice(['NN', 'JJ'])

    noun_suffixes = ['ity', 'ion', 'dom', 'ant', 'ade', 'ess', 'ice', 'ism', 'ist', 'ary', 'ery', 'ory']

    if (previous_state != 'begin_sent' and word[0].isupper()):
                
        if (word[-1] == 's'):
            current_state = 'NNPS'
            
        else:
            current_state = 'NNP'

    elif (word[-1] == 's'):
        current_state = 'NNS'

    elif ('-' in word):

        current_state = random.choice(['NN', 'JJ'])
    
    elif (word.endswith('ed')):

        current_state = random.choice(['VBN', 'VBD'])

    elif (word.endswith('er')):

        current_state = 'JJR'

    elif (word.endswith('ing') or word.endswith('ize')):

        current_state = 'VBG'

    elif (word.endswith('ly')):

        current_state = 'RB'

    elif (word[-3:] in noun_suffixes):

        current_state = 'NN'

    else:

        is_digit = 0
        is_noun = 0
        for l in word:
            if (l.isdigit()):
                is_digit = 1
            if (l.isalpha()):
                is_noun = 1

        if (is_digit):
            if(is_noun):
                current_state = 'NNP'
            else:
                current_state = 'CD'

    return current_state

def viterbi(f_out, sentence, likelihood, transition, words_in_training_corpus):

    path = { pos:[] for pos in likelihood}
    current_probabilities = {}

    for state in likelihood:

        if sentence[0] in words_in_training_corpus:

            current_probabilities[state] = likelihood[state].get(sentence[0], 0)*transition['begin_sent'].get(state,0)
        else:
            word_state = OOV(sentence[0], 'begin_sent')
            current_probabilities[state] = .001*transition['begin_sent'].get(word_state, 0)

    for i in range(1, len(sentence)):

        prev_probabilities = current_probabilities
        current_probabilities = {}

        if sentence[i] in words_in_training_corpus:

            for curr_pos in likelihood:
                max_prob, previous_pos = max(((prev_probabilities[las_pos]*likelihood[curr_pos].get(sentence[i], 0)*transition[las_pos].get(curr_pos,0), las_pos) for las_pos in likelihood))

                current_probabilities[curr_pos] = max_prob
                path[curr_pos].append(previous_pos)
        else:

            for curr_pos in likelihood:
                max_prob, previous_pos = max(((prev_probabilities[las_pos]*.001*transition[las_pos].get(OOV(sentence[i], las_pos),0), las_pos) for las_pos in likelihood))

                current_probabilities[curr_pos] = max_prob
                path[curr_pos].append(previous_pos)

    max_score = -1
    last_state = ''

    for state in current_probabilities:
        if current_probabilities[state] > max_score:
            max_score = current_probabilities[state]
            last_state = state

    max_path = [last_state]

    for i in range(len(sentence) - 2, -1, -1):
        tag = path[last_state][i]
        last_state = tag
        max_path = [tag]+ max_path

    for i,s in enumerate(max_path):
        f_out.write(sentence[i] + '\t' + s + '\n')
    f_out.write('\n')



def sentence(fn_in, fn_out, likelihood, transition, words_in_training_corpus):

    with open(fn_in, 'r') as f:

        f_out = open(fn_out, 'w')

        sentence = []

        for line in f:

            if (line.strip() == ''):
                viterbi(f_out, sentence, likelihood, transition, words_in_training_corpus)
                sentence = []

            else:
                sentence.append(line.strip())

        f_out.close()


words_in_training_corpus = []
likelihood = {}
transition = {}

if (len(sys.argv) != 4):
    print()
    print('incorrect usage: must include 3 filenames in arguments\n[training file | input file | desired output file]')
    print()
    exit()

ftrain = sys.argv[1]
fin = sys.argv[2]
fout = sys.argv[3]

training(ftrain, likelihood, transition, words_in_training_corpus)
sentence(fin, fout, likelihood, transition, words_in_training_corpus)




