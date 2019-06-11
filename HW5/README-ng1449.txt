This program implements a noun group tagger using various features

To run this program you will need to do the following in commandline:

$ python3 HW5-ng1449.py WSJ_02-21.pos-chunk training.feature WSJ_23.pos test.feature

With this program, I implemented the following features:

- current word
- current POS
- previous word
- previous POS
- second previous word
- second previous POS
- next word
- next POS
- second next word
- second next POS
- proper nouns
- proper noun groups
- noun groups
- money
- dates
- location
- abbreviations

I would've included additional features to the system, however, it would take too long and there were too many errors occuring. Futhermore, there were many formatting errors I could not get passed. 

With that I achieved a 96.53% accuracy, 90.42% precision, 92.23% recall and 91.31% f-measure for the WSJ_24.pos development corpus.