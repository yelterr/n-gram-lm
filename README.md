# N-gram Language Model
Done as a project during a Natural Language Processing Course
The original assigment was to create a bi-gram that used _Pride and Prejudice_ as the language model data. I did that, then made a few edits and made a tri-gram language model that used the combined texts of Dostoyevsky as the data. 

### Problems fixed in my version
- The original cut off lines by actual lines instead of sentences themselves. This caused the LM to end sentences with "the" and other words. I solved this problem by splitting lines based on sentences with nltk (in util.py).
- The LM would produce an error whenever it was met with context it hadn't seen before. To solve this, I implemented backoff. If the word was never found in any context, I chose a random word (in A2_Etha_n.py).

### Files
- A2_Etha_n.ipynb: A jupyter notebook file showing the actual running of the n-gram and some predictions at the bottom.
- A2_Etha_n.py: The various functions needed to train the n-gram and predict words
- util.py: The functions used to load the data and tokenize the sentences that were already written before I started the assignment. The 'tokenize_texts()' function is unedited, but the other two functions I've edited to fit the tri-gram.
