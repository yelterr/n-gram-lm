import os.path
import re
import spacy
from tqdm import tqdm
import requests

nlp = spacy.load("en_core_web_sm")


def tokenize_texts(texts, lowercase=True):
    """
    Tokenize texts
    [Input] texts (list of string): sentences e.g., ["I am Bob.", "I like to play baseball.", "......"]
    [Output] tokenized_texts (list of list of string) e.g., [["I","am","Bob","."] , ["I","like","to","play", "baseball", "."], ...]
    """
    print("Tokenizing......")
    tokenized_texts = []
    for text in tqdm(texts):
        # Do some cleaning
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r'\s+', ' ', text)

        # Do case folding to lowercase if specified
        if lowercase == True:
            text = text.lower()

        # Tokenize sentence and return list of tokens
        tokenized_text = [t.text.strip() for t in nlp(text) if t.text.strip() != '']
        if tokenized_text:
            tokenized_texts.append(tokenized_text)
    return tokenized_texts


def prepare_corpus(save_path):
    """
    Download data resource `Pride and prejudice` into the local dir.
    [Input] save_path (string) store path
    """
    # create target dir if it is not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
    # Download the file using the given URL
    r = requests.get(url, allow_redirects=True)
    # Specify where to save the file
    file_name = os.path.join(save_path, 'pride_and_prejudice.txt')
    # Save the file locally
    open(file_name, 'wb').write(r.content)


def get_corpus(path):
    """
    Read the file
    [Input] path (str): path to the file, e.g., "./data/pride_and_prejudice.txt"
    [Output] texts (list of string): a list contains sentences, e.g., ["I am Bob.", "I like to play baseball.", "......"]
    """
    texts = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                texts.append(line)
    print("Total {} lines in \"{}\":".format(len(texts), path.split("/")[-1].strip()))
    return texts
