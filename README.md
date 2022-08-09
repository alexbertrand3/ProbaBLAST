# ProbaBLAST

A modified BLAST algorithm for querying nucleotide sequences on probabilistic target databases.

Read the full report [here](https://github.com/alexbertrand3/ProbaBLAST/blob/main/ProbaBLAST.pdf).

## Usage

### Option 1 - Google Colab (Recommended)
Open the project here:
https://colab.research.google.com/github/alexbertrand3/ProbaBLAST/

Run the code blocks as you would any other notebook.

### Option 2 - Local Machine
Clone the repo; all of the python files required for use are located in the src folder. A full analysis can be performed by running testing.py (it will take a while). To run specific analyses, just comment out the ones you don't want (a better way of running locally is coming soon&trade;).

### Running on New Data
Replace data/sequence.txt and data/confidence.txt with your data. See existing data for formatting.

A new sequence library will need to be built, for which you may want to change the word length (w) and probability threshold (p) settings. You may want to change the filename of the library.

## Contact
Alex Bertrand - alexander.bertrand@mail.mcgill.ca or alexbertrand3@gmail.com

## License
Distributed under the GNU LESSER GENERAL PUBLIC LICENSE. See LICENSE for more information.

## Acknowledgements
Yanlin Zhang &ndash; for devising this problem and providing data to test with.
