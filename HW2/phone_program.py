import re
import sys

if(len(sys.argv) != 2):
	print()
	print("Incorrect Usage: must include one file name in commandline.")
	print()

filename = sys.argv[-1]

def read_txt(filename):

	def regexp(filename):

		matches = []

		with open(filename, 'r') as f:

			p1 = r'\s((\+?\s?\d{0,2}(\s|\-)?\(?\d{3}\)?\s?)|(\(?\d{3}\)?\s?))?(\s|\-|\.)?\d{3}(\s|\-|\.)?\d{4}'

			pattern = re.compile(p1)

			for line in f:

				pos = 0

				match = pattern.search(line, pos)

				while(match):

					matches.append(match.group(0))
					pos = match.endpos
					match = pattern.search(line, pos)

		return matches


	try:

		return regexp(filename)

	except IOError:

		print('An IOError has occurred.')

	except:

		print('An unexpected error has occurred.')


matches = read_txt(filename)
f = open("telephone_output.txt", 'w')

for item in matches:

	f.write(item.strip() + '\n')

f.close()