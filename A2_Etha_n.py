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
        self.ngram_counter = defaultdict(int)

        # Dictionary that keeps list of candidate words given context
        # When generating a text, we only pick from those candidate words
        self.context = {}

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
        ngrams = []
        for words in padding_texts:
            words = self.vocab_lookup(words)
            #########################################################################################
            ### Your code starts here ###############################################################

            sentence = words # Makes more sense in my brain
            for i, word in enumerate(sentence):
                if i == 0:
                    continue
                else:
                    ngrams.append((sentence[i-1], word))

            ### Your code ends here #################################################################
            #########################################################################################
        if not ngrams:
            print("Warning!!! You need to implement this function! This function accounts for 20 points!")
        return ngrams

    def fit(self, ngrams):
        """
        Train N-gram Language Models.
        [Input] ngrams (list of tuples) e.g., [("<s>", "I"), ("I", "am"), .....] for bi-gram
        """
        self.ngram_counter = defaultdict(int)
        self.context = {}
        for ngram in ngrams:
            prev_words, target_word = ngram
            #########################################################################################
            ### Your code starts here ###############################################################
            # Tips: self.ngram_counter is used for keeping track of how many times ngram has appeared in the text before.
            # e.g., {('admire', 'you'): 1, ('you', 'much'): 3}
            # Tips: self.context is a Dictionary that keeps list of candidate words given context.
            # e.g., {'revenue': ['service'], 'ein': ['or'], 'federal': ['tax', 'laws']}
            
            if ngram not in self.ngram_counter.keys():
                self.ngram_counter[ngram] = 1
            else:
                self.ngram_counter[ngram] += 1
                
            if prev_words not in self.context.keys():
                self.context[prev_words] = [target_word]
            else:
                self.context[prev_words].append(target_word)

            ### Your code ends here #################################################################
            #########################################################################################
        if not self.context:
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
            #########################################################################################
            ### Your code starts here ###############################################################
            # Tips: calculate count(Wn-1, Wn)
            # Tips: calculate count(Wn-1)
            # Tips: calculate Padd-k(Wn|Wn-1) and remember add-k smoothing here.

            ngram_count = self.ngram_counter[(context, token)]
            context_count = len(self.context[context])

            result = (ngram_count + 0.75) / (context_count + 0.75 * self.vocabulary_size)
            
            ### Your code ends here #################################################################
            #########################################################################################
        except KeyError:
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
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tips: Get all candidate words for the given context via self.context
        # Tips: Get the probabilities for each ngram (context+word) via self.calc_prob
        # you may store all probabilities in np.array
        # Tips: Return a random candidate word based on the probability distribution
        # you may use np.random.choice

        candidates = self.context[context]
        probabilities = np.zeros((len(candidates),))

        for i, candidate in enumerate(candidates):
            probabilities[i] = self.calc_prob(context, candidate)
        
        # The below did not work for me since my probabilities do not sum up to 1.
        # I'm not sure whether this is an error on my part or whether add-k probabilities
        # normally don't add up to 1. Anyways, I've just made it so that they do.
        # selected_token = np.random.choice(candidates, p=probabilities)

        new_probabilities = probabilities/sum(probabilities)
        selected_token = np.random.choice(candidates, p=new_probabilities)
        
        ### Your code ends here #################################################################
        #########################################################################################
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
        if not self.context:
            print("Warning!!! You need to implement this function! This function accounts for 10 points!")
            return

        n = self.n

        start_context = start_context.split()
        
        # The following block merely prepares the first context; note that the context is always of size
        # (self.n - 1) so depending on the start_context (representing the start/seed words), we need to
        # pad or cut off the start_context.
        if len(start_context) == (n - 1):
            context_queue = start_context.copy()
        elif len(start_context) < (n - 1):
            context_queue = ((n - (len(start_context) + 1)) * [self.sos]) + start_context.copy()
        elif len(start_context) > (n - 1):
            context_queue = start_context[-(n - 1):].copy()
        result = start_context.copy()
        
        # The main loop for generating words
        for _ in range(token_count):
            # Generate the next token given the current context
            obj = self.random_token(" ".join(context_queue))
            # Add generated word to the result list
            result.append(obj)
            # Remove the first token from the context
            context_queue.pop(0)
            if obj == self.eos:
                # If we generate the EOS token, we can return the sentence (without the EOS token)
                return ' '.join(result[:-1])
                #context_queue.append(self.sos) # If I want sentences to keep going further.
            else:
                # Otherwise create the new context and keep generate the next word
                context_queue.append(obj)
        # Fallback if we predict more than token_count tokens
        return ' '.join(result)
