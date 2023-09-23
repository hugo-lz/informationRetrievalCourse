from gensim import corpora
from gensim import models
from gensim import similarities

from gensim_demo import index
import sys

def search(index_folder, query):

    dictionary = corpora.Dictionary.load(index.get_dictionary_file_name(index_folder))

    query_document = index.generate_terms(query)
    print('query words: ', query_document)
    query_bow = dictionary.doc2bow(query_document)
    print('query bow: ', query_bow)

    index_matrix = similarities.MatrixSimilarity.load(index.get_index_file_name(index_folder))
    model = models.TfidfModel.load(index.get_model_file_name(index_folder))

    print('query tfidf vector: ',model[query_bow])
    sims = index_matrix[model[query_bow]]

    print('Returned documents:')
    for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        print(document_number, score)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    index_folder = '../index'
    i = 1
    while (i < len(sys.argv)):
        if sys.argv[i] == '-index':
            index_folder = sys.argv[i+1]
            i = i + 1
        elif sys.argv[i] == '-language':
            # -language is expected to be either 'english' or 'spanish'
            index.LANGUAGE = sys.argv[i + 1]
            i = i + 1
        i = i + 1

    #query = 'system engineering'
    query = input('Introduce a query: ')
    search(index_folder, query)