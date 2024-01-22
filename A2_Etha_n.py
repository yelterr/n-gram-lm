import numpy as np
from collections import defaultdict


class MyNgramLM(object):

    def __init__(self, n, k, sos='<s>', eos='</s>'):
        self.n, self.k, self.sos, self.eos = n, k, sos, eos

        self.unk = "[UNK]"

        # We need at least bigrams for this model to work
        if self.n < 2:
            raise Exception('Size of n-grams must be at least 2!')

        # keeps track of how many times ngram has appeared in the text before
        self.trigram_counter = defaultdict(int)
        self.bigram_counter = defaultdict(int)

        # Dictionary that keeps list of candidate words given context
        # When generating a text, we only pick from those candidate words
        self.tri_context = {}
        self.bi_context = {}

        self.vocabulary = set()
        self.vocabulary_size = 0
        print(f"Initialize a {self.n}-gram model with add-{self.k}-smoothing.")

    def add_padding(self, tokenized_texts):
        """
        Padding texts
        [Input] tokenized_texts (list of list of string) e.g., [["I","am","Bob","."] , ["I","like","to","play", "baseball", "."], ...]
        [Output] padding_texts (list of list of string) e.g., [["<s>", "I","am","Bob","." , "</s>"] , ["<s>", "I","like","to","play", "baseball", "." "</s>"], ..
        """
        padding_texts = []
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tips: you need to use self.sos and self.eos as padding tokens for start and end positions respectively.
        
        for sentence in tokenized_texts:
            sentence = [self.sos] + sentence + [self.eos]
            padding_texts.append(sentence)

        ### Your code ends here #################################################################
        #########################################################################################
        if not padding_texts:
            print("Warning!!! You need to implement this function! This function accounts for 10 points!")
        return padding_texts

    def vocab_lookup(self, sequence):
        """
        Look up one sentence based on the vocabulary.
        [Input] sequence (string or list ) e.g, "I am Bob ." or ["I", "am", "Bob", "."]
        [Output] output (string or list) e.g, (for string input) If all the words are in the vocab, return "I am Bob ." Otherwise, 'Bob' is not in the vocab, return "I am [UNK] ."
        """
        output = None
        if isinstance(sequence, str):
            output = " ".join(
                [word.strip() if word.strip() in self.vocabulary else self.unk for word in sequence.split()]).strip()
        elif isinstance(sequence, list):
            output = [word.strip() if word.strip() in self.vocabulary else self.unk for word in sequence]
        return output

    def build_vocabulary(self, texts, cutoff_freq):
        """
        Build vocabulary
        [Input] texts (list of list of string) e.g., [["<s>", "I","am","Bob","." , "</s>"] , ["<s>", "I","like","to","play", "baseball", "." "</s>"], ..
                cutoff_freq (int) Only words with frequencies above the cutoff_freq were retained in the vocab. e.g., 5
        """
        vocabulary = set()
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tips: You can use a dictionary object to record the frequency of each word.
        # Tips: For words with frequencies above the cutoff_freq, you can store them in the vocabulary object.
        
        freq_dict = {}
        for sentence in texts:
            for word in sentence:
                if word not in freq_dict.keys():
                    freq_dict[word] = 1
                else:
                    freq_dict[word] += 1

        for word, freq in freq_dict.items():
            if freq > cutoff_freq:
                vocabulary.add(word)
                    
        ### Your code ends here #################################################################
        #########################################################################################
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        if vocabulary:
            print("Vocab size:", self.vocabulary_size)
        else:
            print("Warning!!! You need to implement this function! This function accounts for 10 points!")

    def get_ngrams(self, padding_texts):
        """
        Returns ngrams of the given padding_texts
        [Input] padding_texts (list of list of string)  e.g., [["<s>", "I","am","Bob","." , "</s>"] , ["<s>", "I","like","to","play", "baseball", "." "</s>"], ..
        [output] ngrams (list of tuples) e.g., [("<s>", "I"), ("I", "am"), .....] for bi-gram
        """
        trigrams = []
        bigrams = []
        for words in padding_texts:
            words = self.vocab_lookup(words)
            #########################################################################################
            ### Your code starts here ###############################################################

            # For the tri-grams
            sentence = words # Makes more sense in my brain
            for i, word in enumerate(sentence):
                if i == 0 or i == 1:
                    continue
                else:
                    trigrams.append((sentence[i-2], sentence[i-1], word))

            # For the BI-grams
            for i, word in enumerate(sentence):
                if i == 0:
                    continue
                else:
                    bigrams.append((sentence[i-1], word))

            ### Your code ends here #################################################################
            #########################################################################################
        if not trigrams:
            print("Warning!!! You need to implement this function! This function accounts for 20 points!")
        return (trigrams, bigrams)

    def fit(self, trigrams, bigrams):
        """
        Train N-gram Language Models.
        [Input] ngrams (list of tuples) e.g., [("<s>", "I"), ("I", "am"), .....] for bi-gram
        """
        self.ngram_counter = defaultdict(int)
        self.tri_context = {}
        self.bi_context = {}
        
        # Building the trigrams data
        for trigram in trigrams:
            prev_prev_word, prev_word, target_word = trigram

            if trigram not in self.trigram_counter.keys():
                self.trigram_counter[trigram] = 1
            else:
                self.trigram_counter[trigram] += 1
                
            if (prev_prev_word, prev_word) not in self.tri_context.keys():
                self.tri_context[(prev_prev_word, prev_word)] = [target_word]
            else:
                self.tri_context[(prev_prev_word, prev_word)].append(target_word)

        # Building the BIgrams data
        for bigram in bigrams:
            prev_word, target_word = bigram

            if bigram not in self.bigram_counter.keys():
                self.bigram_counter[bigram] = 1
            else:
                self.bigram_counter[bigram] += 1
                
            if prev_word not in self.bi_context.keys():
                self.bi_context[prev_word] = [target_word]
            else:
                self.bi_context[prev_word].append(target_word)
        
        if not self.tri_context:
            print("Warning!!! You need to implement this function! This function accounts for 20 points!")
        else:
            print("Finish Language Model Training.")

    def calc_prob(self, context, token):
        """
        Calculates probability of a token given a context
        [Input] context (string) e.g., "I"
                token (string) e.g., "am"
        [output] result (float) conditional probability
        """
        try:
            result = None
            # Now I need to pass in either 2 context words in a tuple to this function or just one in a string
            prev_prev_word, prev_word = context

            tri_gram_count = self.trigram_counter[(prev_prev_word, prev_word, token)]
            tri_context_count = len(self.tri_context[context])

            result = (tri_gram_count + 0.75) / (tri_context_count + 0.75 * self.vocabulary_size)
        except:
            # If the context provided is not in the tri-gram context, or not a tuple
            try:            
                bi_gram_count = self.bigram_counter[(context, token)]
                bi_context_count = len(self.bi_context[context])
    
                result = (bi_gram_count + 0.75) / (bi_context_count + 0.75 * self.vocabulary_size)
            
            except:
                # Not in the tri-gram or bi-gram data
                result = 0.0

        if result is None:
            print("Warning!!! You need to implement this function! This function accounts for 20 points!")
        return result

    def random_token(self, context):
        """
        Given a context we "semi-randomly" select the next word to append in a sequence
        [Input] context (string) e.g., "i"
        [Output]
        """
        selected_token = None

        # Dropout when choosing next word
        try:
            candidates = self.tri_context[context]
            probabilities = np.zeros((len(candidates),))
    
            for i, candidate in enumerate(candidates):
                probabilities[i] = self.calc_prob(context, candidate)

            new_probabilities = probabilities/sum(probabilities)
            selected_token = np.random.choice(candidates, p=new_probabilities)
        except:
            # Since tri-grams didn't work, now time to try with bi-grams
            try:
                context = context[-1]
                candidates = self.bi_context[context]
                probabilities = np.zeros((len(candidates),))
        
                for i, candidate in enumerate(candidates):
                    probabilities[i] = self.calc_prob(context, candidate)
    
                new_probabilities = probabilities/sum(probabilities)
                selected_token = np.random.choice(candidates, p=new_probabilities)
            
            # bi-gram doesn't exist either, so time to return a random word.
            except:
                # I typecast self.vocabulary to a list so that I can use random choice on it
                selected_token = np.random.choice(list(self.vocabulary))
        
        if selected_token is None:
            print("Warning!!! You need to implement this function! This function accounts for 10 points!")
        return selected_token

    def generate_text(self, token_count: int, start_context: str):
        """
        Iteratively generates a sentence by predicted the next word step by step
        [Input] token_count (int): number of words to be produced
                start_context (string): start words
        [Output] generated text (string)
        """
        n = self.n

        start_context = start_context.split()
        
        # The following block merely prepares the first context; note that the context is always of size
        # (self.n - 1) so depending on the start_context (representing the start/seed words), we need to
        # pad or cut off the start_context.

        # I made the context_queue a tuple so that it could be input into context
        if len(start_context) == (n - 1):
            context_queue = tuple(start_context.copy())
        elif len(start_context) < (n - 1):
            context_queue = tuple(((n - (len(start_context) + 1)) * [self.sos]) + start_context.copy())
        elif len(start_context) > (n - 1):
            context_queue = tuple(start_context[-(n - 1):].copy())
        result = start_context.copy()
        
        # The main loop for generating words
        for _ in range(token_count):
            # Generate the next token given the current context
            obj = self.random_token(context_queue)
            # Add generated word to the result list
            result.append(obj)
            # Remove the first token from the context
            context_queue = context_queue[1:]
            if obj == self.eos:
                # If we generate the EOS token, we can return the sentence (without the EOS token)
                return ' '.join(result[:-1])
                #context_queue = (context_queue[0], self.sos) # If I want sentences to keep going further.
            else:
                # Otherwise create the new context and keep generate the next word
                context_queue = (context_queue[0], obj)
                
        # Fallback if we predict more than token_count tokens
        # Commented out for loop is only needed if we continue on past the end of the predicted sentence.
        '''
        # Replace the end-of-sentence symbol with nothing so that the output is better.
        for i, word in enumerate(result):
            if word == self.eos:
                result[i] = ""
        '''
        return ' '.join(result)
