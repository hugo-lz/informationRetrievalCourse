"""
index.py
Author: Raúl Soler y Hugo López
Last update: 2025-9-16

Simple program to create an inverted index with the contents of text/xml files contained in a docs folder
This program is based on the whoosh library. See https://pypi.org/project/Whoosh/ .
Usage: python index.py -index <index folder> -docs <docs folder>
"""

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter, Filter
from nltk.stem.snowball import SnowballStemmer

import os

import sys

import datetime

import xml.etree.ElementTree as ET

def create_folder(folder_name):
    if (not os.path.exists(folder_name)):
        os.mkdir(folder_name)

class SnowballFilter(Filter):
    def __init__(self, language='spanish'):
        self.stemmer = SnowballStemmer(language)

    def __call__(self, tokens):
        for t in tokens:
            t.text = self.stemmer.stem(t.text)
            yield t

spanish_analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter(lang="es") #| SnowballFilter('spanish')

class MyIndex:
    def __init__(self,index_folder):
        # Para poder filtrar por autor, director, departamento al que se adscribe el trabajo,
        #  términos contenidos en el título, descripción o palabras clave  y año de defensa
        schema = Schema(path=ID(stored=True), 
                        content=TEXT(analyzer=spanish_analyzer, stored=True), 
                        title=TEXT(analyzer=spanish_analyzer, stored=True), 
                        subject=TEXT(analyzer=spanish_analyzer, stored=True),
                        type_of_work=TEXT(analyzer=spanish_analyzer, stored=True),
                        author=TEXT(analyzer=spanish_analyzer, stored=True),
                        director=TEXT(analyzer=spanish_analyzer, stored=True),
                        date=ID(stored=True),
                        department=TEXT(analyzer=spanish_analyzer, stored=True),
                        description=TEXT(analyzer=spanish_analyzer, stored=True),
                        identifier=ID(stored=True),
                        modified=STORED
        )
                
        create_folder(index_folder)
        index = create_in(index_folder, schema)
        self.writer = index.writer()

    def index_docs(self,docs_folder):
        if (os.path.exists(docs_folder)):
            for file in sorted(os.listdir(docs_folder)):
                # print(file)
                if file.endswith('.xml'):
                    self.index_xml_doc(docs_folder, file)
                elif file.endswith('.txt'):
                    self.index_txt_doc(docs_folder, file)
        self.writer.commit()

    def index_txt_doc(self, foldername,filename):
        file_path = os.path.join(foldername, filename)
        # print(file_path)
        with open(file_path) as fp:
            text = ' '.join(line for line in fp if line)
        # print(text)
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%a, %d %b %Y %H:%M:%S +0000')
        self.writer.add_document(path=filename, content=text, modified=modified )

    def index_xml_doc(self, foldername, filename):
        file_path = os.path.join(foldername, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        # título del trabajo
        title = ' '.join([elem.text.strip() for elem in root.findall('.//{*}title') if elem.text])
        # tipo de trabajo (TFG, TFM, Tesis Doctoral...)
        type_of_work = ' '.join([elem.text.strip() for elem in root.findall('.//{*}type') if elem.text])
        # puede haber varios directores
        director= ' '.join([elem.text.strip() for elem in root.findall('.//{*}contributor') if elem.text])
        # autor de la tesis
        author = ' '.join([elem.text.strip() for elem in root.findall('.//{*}creator') if elem.text])
        # fecha de defensa de la tesis
        date = ' '.join([elem.text.strip() for elem in root.findall('.//{*}date') if elem.text])
        # departamento al que se adscribe el trabajo
        department = ' '.join([elem.text.strip() for elem in root.findall('.//{*}publisher') if elem.text])
        # temas que trata el trabajo
        subject = ' '.join([elem.text.strip() for elem in root.findall('.//{*}subject') if elem.text])
        # descripción o palabras clave
        description = ' '.join([elem.text.strip() for elem in root.findall('.//{*}description') if elem.text])
        # identificador del trabajo
        identifier = ' '.join([elem.text.strip() for elem in root.findall('.//{*}identifier') if elem.text])

        raw_text = "".join(root.itertext())
        text = ' '.join(line.strip() for line in raw_text.splitlines() if line)
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%a, %d %b %Y %H:%M:%S +0000')
        self.writer.add_document(
            path=filename,
            content=text,
            title=title,
            type_of_work=type_of_work,
            author=author,
            director=director,
            subject=subject,
            description=description,
            date=date,
            department=department,
            identifier=identifier,
            modified=modified
        )

if __name__ == '__main__':

    index_folder = '../whooshindex'
    docs_folder = '../docs'
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-index':
            index_folder = sys.argv[i + 1]
            i = i + 1
        elif sys.argv[i] == '-docs':
            docs_folder = sys.argv[i + 1]
            i = i + 1
        i = i + 1

    my_index = MyIndex(index_folder)
    my_index.index_docs(docs_folder)