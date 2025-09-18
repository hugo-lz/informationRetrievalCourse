"""
search.py
Author: Raúl Soler y Hugo López
Last update: 2025-9-16

Program to search a free text query on a previously created inverted index.
This program is based on the whoosh library. See https://pypi.org/project/Whoosh/ .
Usage: python search.py -index <index folder>
"""

import sys

from whoosh.qparser import QueryParser
from whoosh.qparser import OrGroup
from whoosh import scoring
from whoosh.qparser import MultifieldParser
from nltk.stem.snowball import SnowballStemmer
import whoosh.index as index



class MySearcher:
    def __init__(self, index_folder, model_type = 'tfidf', info=False):
        ix = index.open_dir(index_folder)
        if model_type == 'tfidf':
            # Apply a vector retrieval model as default
            self.searcher = ix.searcher(weighting=scoring.TF_IDF())
        else:
            # Apply the probabilistic BM25F model, the default model in searcher method
            self.searcher = ix.searcher()
        fields = ["author", "director", "department", "title", "description", "subject", "date", "content"]
        self.parser = MultifieldParser(fields, ix.schema, group=OrGroup)
        self.info = info

    def search(self, query_text, limit=None):
        query = self.parser.parse(query_text)
        return self.searcher.search(query, limit=limit)

    def print_results(self, results):
        print("Returned documents:")
        for i, result in enumerate(results, start=1):
            print(f'{i} - File path: {result.get("path")}, Similarity score: {result.score}')
            if self.info:
                print(f'\t Modified : {result.get("modified")}')
            ###
            # if self.info:
            # print(f'Modified : {result.get("modified")}')
            # print(f'Title: {result.get("title")}')
            # print(f'Author: {result.get("author")}')
            # print(f'Director(s): {result.get("director")}')
            # print(f'Department: {result.get("department")}')
            # print(f'Description: {result.get("description")}')
            # print(f'Subject: {result.get("subject")}')
            # print(f'Date: {result.get("date")}')
            # print(f'Type of work: {result.get("type_of_work")}')
            ###

if __name__ == '__main__':
    info = False
    index_folder = '../whooshindex'
    info_needs_file = None
    output_file = None

    i = 1
    while (i < len(sys.argv)):
        if sys.argv[i] == '-index':
            index_folder = sys.argv[i+1]
            print(f'Index folder: {index_folder}')
            i = i + 1
        if sys.argv[i] == '-infoNeeds':
            # Leer el contenido del archivo determinado por InfoNeeds para obtener la consulta 
            info_needs_file = sys.argv[i + 1]
            i = i + 1
        if sys.argv[i] == '-output':
            output_file = sys.argv[i + 1]
            print(f'Output file: {output_file}')
            i = i + 1
        if sys.argv[i] == '-info':
            info = True
        i = i + 1

    searcher = MySearcher(index_folder=index_folder, info=info)

    if info_needs_file and output_file:
        with open(info_needs_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        with open(output_file, 'w') as f_out:
            for qid, query in enumerate(queries, start=1):
                results = searcher.search(query, limit=100)
                for r in results:
                    f_out.write(f"{qid}\t{r.get('identifier')}\n")
    else: 
        query = input("Introduce a query: ")
        while query != 'q':
            results = searcher.search(query)
            searcher.print_results(results)
            query = input("Introduce a query ('q' for exit): ")