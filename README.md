# N-gram Language Model
Done as a project during a Natural Language Processing Course
The original assigment was to create a bi-gram that used _Pride and Prejudice_ as the language model data. I did that, then made a few edits and made a tri-gram language model that used the combined texts of Dostoyevsky as the data. 

### Problems fixed in my version
- The original cut off lines by actual lines instead of sentences themselves. This caused the LM to end sentences with "the" and other words. I solved this problem by splitting lines based on sentences with nltk (in util.py).
- The LM would produce an error whenever it was met with context it hadn't seen before. To solve this, I implemented backoff. If the word was never found in any context, I chose a random word (in A2_Etha_n.py).
