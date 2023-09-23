"""
index.py
Author: Javier Nogueras Iso
Last update: 2023-09-22

Program to create a term-document matrix index with either a vector model (tf-idf) or OkapiBM25 model.
This program is based on the gensim Python library. See https://github.com/RaRe-Technologies/gensim/#documentation .
Usage: python index.py -docs <doc folder> -index <index folder> -language <english|spanish>
"""

import os
import pprint
import sys
import xml.etree.ElementTree as ET

from gensim import corpora
from gensim import models
from gensim import similarities
from nltk.stem.snowball import SnowballStemmer

LANGUAGE = 'english'
#LANGUAGE = 'spanish'

STOP_LIST = None

def create_folder(folder_name):
    if (not os.path.exists(folder_name)):
        os.mkdir(folder_name)


def get_dictionary_file_name(folder_name):
    create_folder(folder_name)
    return os.path.join(folder_name, 'dictionary')


def get_model_file_name(folder_name):
    create_folder(folder_name)
    return os.path.join(folder_name, 'model')


def get_index_file_name(folder_name):
    create_folder(folder_name)
    return os.path.join(folder_name, 'index')

def apply_stemming(words):
    # the stemmer requires a language parameter
    snow_stemmer = SnowballStemmer(language=LANGUAGE)

    # stem's of each word
    stem_words = []
    for w in words:
        x = snow_stemmer.stem(w)
        stem_words.append(x)

    # print stemming results
    #for e1, e2 in zip(words, stem_words):
    #    print(e1 + ' ----> ' + e2)

    return stem_words

def get_stop_list():
    global STOP_LIST
    if STOP_LIST is not None:
        return STOP_LIST
    else:
        if LANGUAGE == 'english':
            STOP_LIST = set('for a of the and to in'.split(' '))
        elif LANGUAGE == 'spanish':
            STOP_LIST = set('para un una unos unas de el la lo los las y a en'.split(' '))
        return STOP_LIST

def generate_terms(text, stemming=True):
    stoplist = get_stop_list()

    word_vector = [word for word in text.lower().split() if word not in stoplist]

    if stemming:
        #apply stemming
        word_vector = apply_stemming(word_vector)

    return word_vector

def normalize(word):
    x = ",;:.-/\\(){}[]¿?¡!\"#&'+*%$_"
    y = "                          "
    table = str.maketrans(x, y)
    return word.translate(table).strip()

def process_text_file(foldername, filename):
    file_path = os.path.join(foldername, filename)
    # print(file_path)
    with open(file_path) as fp:
        text = ' '.join(normalize(line) for line in fp if line)
    # print(text)
    return text


def process_xml_file(foldername, filename):
    file_path = os.path.join(foldername, filename)
    #print(file_path)

    tree = ET.parse(file_path)
    root = tree.getroot()
    raw_text = "".join(root.itertext())
    #print(raw_text)
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in raw_text.splitlines())
    # remove punctuation
    text = ' '.join(normalize(chunk) for line in lines for chunk in line.split())
    # print(text)
    return text

class MyCorpus:
    def __init__(self, folder_name):
        self.folder_name = folder_name

    def __iter__(self):
        for file in sorted(os.listdir(self.folder_name)):
            if file.endswith('.xml') > 0:
                text = process_xml_file(self.folder_name, file)
            else:
                text = process_text_file(self.folder_name, file)
            yield generate_terms(text)

def get_example_corpus():
    text_corpus = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

    stoplist = get_stop_list()

    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in text_corpus]

    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    pprint.pprint(processed_corpus)

    return processed_corpus

def create_dictionary(processed_corpus, compact=True):
    dictionary = corpora.Dictionary(processed_corpus)

    if compact:
        # remove words that appear only once
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
        dictionary.filter_tokens(once_ids)  # remove words that appear only once
        dictionary.compactify()  # remove gaps in id sequence after words that were removed

    # print(dictionary)
    return dictionary

def create_index(index_folder, docs_folder, model_type='tfidf'):
    # processed_corpus = get_example_corpus()
    processed_corpus = MyCorpus(docs_folder)
    # for vector in processed_corpus:  # load one vector into memory at a time
    #    print(vector)

    dictionary = create_dictionary(processed_corpus)
    length = len(dictionary.token2id)
    print('Dictionary length: ', length)
    pprint.pprint(dictionary.token2id)

    dictionary_file_name = get_dictionary_file_name(index_folder)
    dictionary.save(dictionary_file_name)

    # new_doc = "Human computer interaction"
    new_doc = "system system minors"
    print('Example document: ', new_doc)
    new_doc_words = generate_terms(new_doc)
    print('Example document words: ', new_doc_words)
    new_vec = dictionary.doc2bow(new_doc_words)
    print('Example document as bow vector: ', new_vec)

    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    # pprint.pprint(bow_corpus)

    # train the model
    if model_type == 'tfidf':
        model = models.TfidfModel(bow_corpus, smartirs='lfc')
    elif model_type == 'okapi':
        model = models.OkapiBM25Model(bow_corpus)
    else:
        print('Model type not recognized')
        exit(1)
    model_file_name = get_model_file_name(index_folder)
    model.save(model_file_name)

    # transform the "system system minors" string
    print('Example document as tfidf vector ',model[new_vec])

    index = similarities.SparseMatrixSimilarity(model[bow_corpus], num_features=length)
    index_file_name = get_index_file_name(index_folder)
    index.save(index_file_name)


if __name__ == '__main__':

    index_folder = '../index'
    docs_folder = '../docs'
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-index':
            index_folder = sys.argv[i + 1]
            i = i + 1
        elif sys.argv[i] == '-docs':
            docs_folder = sys.argv[i + 1]
            i = i + 1
        elif sys.argv[i] == '-language':
            # -language is expected to be either 'english' or 'spanish'
            LANGUAGE = sys.argv[i + 1]
            i = i + 1
        i = i + 1

    create_index(index_folder, docs_folder, 'tfidf')
