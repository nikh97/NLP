fn = 'WSJ_02-21.pos'
tags = []

with open(fn, 'r') as f:

	for line in f:
		if (line.strip() != ''):
			data = line.strip().split('\t')
			##print(line)
			tag = data[0]

			if tag not in tags:
				tags.append(tag)

##print(tags)
print(len(tags))


# def training(filename, likelihood, transition):

# 	with open(filename, 'r') as f:

# 		previous_state = ""

# 		first_line = f.readline()
# 		first_line_arr = first_line.strip().split("\t")

# 		likelihood[first_line_arr[1]] = {first_line_arr[0]: 1}
# 		transition['begin_sent'] = {}
# 		previous_state = 'begin_sent'

# 		for line in f:

# 			if (line.strip() == ''):

# 				if previous_state not in transition:

# 					transition[previous_state] = {'end_sent': 1}

# 				else:

# 					if 'end_sent' not in transition[previous_state]:
# 						transition[previous_state]['end_sent'] = 1
# 					else:
# 						transition[previous_state]['end_sent'] += 1

# 				previous_state = 'begin_sent'

# 			else:

# 				line_arr = (line.strip()).split('\t')

# 				line_arr[0] = line_arr[0].lower()

# 				if line_arr[1] not in likelihood:

# 					likelihood[line_arr[1]] = {line_arr[0]: 1}
# 				else:
# 					if line_arr[0] not in likelihood[line_arr[1]]:
# 						likelihood[line_arr[1]][line_arr[0]] = 1
# 					else:	
# 						likelihood[line_arr[1]][line_arr[0]] += 1


# 				if previous_state not in transition:

# 					transition[previous_state] = {line_arr[1]: 1}
# 				else:
# 					if line_arr[1] not in transition[previous_state]:
# 						transition[previous_state][line_arr[1]] = 1
# 					else:
# 						transition[previous_state][line_arr[1]] += 1

# 				previous_state = line_arr[1]

# 		if previous_state not in transition:

# 			transition[previous_state] = {'end_sent': 1}

# 		else:

# 			if 'end_sent' not in transition[previous_state]:
# 				transition[previous_state]['end_sent'] = 1
# 			else:
# 				transition[previous_state]['end_sent'] += 1


def viterbi(filename, likelihood, transition, words_in_training_corpus):

    tags = []

    with open(filename, 'r') as f:

        max_prev_score = 1
        previous_state = 'begin_sent'
        current_state = 'begin_sent'
        count = 1

        for line in f:

            line = line.strip()

            max_score = 0

            if line == '':

                current_state = 'end_sent'
                max_score = 1
                        
            elif previous_state == 'begin_sent':

                for pos in likelihood:

                    current_score = transition[previous_state].get(pos, 0)

                if (current_score > max_score):
                    max_score = current_score
                    current_state = pos

            else:

                for pos in likelihood:

                    try:
                        current_score = likelihood[pos].get(line, 0) * transition[previous_state].get(pos, 0) * max_prev_score
                    except:
                        print(previous_state)
                        print(pos)

                    if (current_score > max_score):

                        max_score = current_score
                        current_state = pos

            if max_score == 0:

                oov = OOV(line, previous_state, likelihood, transition, count)
                max_prev_score = oov[0]
                previous_state = oov[1]
            else:
                if (current_state == 'end_sent'):
                    previous_state = 'begin_sent'
                else:
                    previous_state = current_state
                max_prev_score = max_score
            count += 1