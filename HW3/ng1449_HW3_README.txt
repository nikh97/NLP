In order to run my program, you will to provide 3 filename arguments ( [ training file | input file | output file] )
$ python3 ng1449_viterbi_HW3.py training.words WSJ_23.words submission.pos

To handle OOV words, I implemented the following rules:

- all OOV words had a likelihood probability of 1/1000 or .0001 for all parts of speech
- all OOV words are assigned either as a NN or JJ randomly as default, which could change depending on the conditions of the word
- If the first letter is capitalized and not the beginning of the sentence, it is assigned a NN or a NNPS (if it ends in 's')
- if not capitalized, but it ends in 's' assign it as a NNS
- if a '-' (hyphen) exists within the word assign it either NN or JJ randomly
- if it ends with 'ed' -> random between VBN, VBD
- if it ends with 'er' -> JJR
- if it ends with 'ing' or 'ize' -> VBG
- if it ends with 'ly' -> RB
- if it ends with 'ity', 'ion', 'dom', 'ant', 'ade', 'ess', 'ice', 'ism', 'ist', 'ary', 'ery', or 'ory' -> NN
- if none of those, check for digits, if the word is alphanumeric assign NNP, if only numeric assign CD

Besides handling OOV words, this system also implements a bigram system to tag the POS. It first takes in the training data and stores likelihood probabilities and transition probabilities in two seperate dictionaries of dictionaries. The system runs viterbi sentence by sentence. Once a sentence is determined, a matrix off backpointer is created as well as a dictionary storing the final scores. Once built, the algorithm backtraces across the matrix to determine the POS tags to be outputted with each word. Using just WSK_02-21.pos as the training file and WSJ_24.pos as the test files, it achieves a 93.82% accuracy rate. After combining WSJ_24.pos with WSJ_02-21.pos, the system jumps to 96.6% accuracy for WSJ_24.words. 

The most difficult part of this implementation came during the coding of viterbi. Despite understanding the algorithm conceptually, it was hard to visualize with the code, data structures to be used, and best way to store and access previous data needed to make calculations. 