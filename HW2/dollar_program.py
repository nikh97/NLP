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

			numbers = r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|quarter|half|-)*'

			p1 = r'(\$(([1-9]\d{0,2}(,\d{3})*)|(([1-9]\d*)?\d))(\.\d+)?(\shundreds?)?(K|\s(thousand|million|billion|trillion))?)'
			p2 = r'((([1-9]\d{0,2}(,\d{3})*)|(([1-9]\d*)?\d))(\shundreds?)?(K|\s(thousand|million|billion|trillion))?((\sdollars?\s(\sand\s)?(\d+\scents?)?)|\scents?\s))'
			p3 = numbers + r'(\shundreds?)?(K|\s(thousand|million|billion|trillion))?((\sdollars?\s(\sand\s)?(\d+\scents?\s)?)|\scents?\s)' 

			patterns = [p1, p2, p3]

			pattern = re.compile('|'.join(x for x in patterns))

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
f = open("dollar_output.txt", 'w')

for item in matches:

	f.write(item.strip() + '\n')

f.close()



