"""
index.py
Author: Javier Nogueras Iso
Last update: 2024-09-07

Simple program to create an inverted index with the contents of text/xml files contained in a docs folder
This program is based on the whoosh library. See https://pypi.org/project/Whoosh/ .
Usage: python index.py -index <index folder> -docs <docs folder>
"""

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import LanguageAnalyzer

import os

import datetime

import xml.etree.ElementTree as ET

def create_folder(folder_name):
    if (not os.path.exists(folder_name)):
        os.mkdir(folder_name)

class MyIndex:
    def __init__(self,index_folder):
        language_analyzer = LanguageAnalyzer(lang="es", expression=r"\w+")
        schema = Schema(path=ID(stored=True), 
                        content=TEXT(analyzer=language_analyzer), 
                        title=TEXT(analyzer=language_analyzer), 
                        subject=TEXT(analyzer=language_analyzer),
                        description=TEXT(analyzer=language_analyzer),
                        modified=STORED)
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
        title = ' '.join([elem.text.strip() for elem in root.findall('.//{*}title') if elem.text])
        subject = ' '.join([elem.text.strip() for elem in root.findall('.//{*}subject') if elem.text])
        description = ' '.join([elem.text.strip() for elem in root.findall('.//{*}description') if elem.text])
        raw_text = "".join(root.itertext())
        text = ' '.join(line.strip() for line in raw_text.splitlines() if line)
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%a, %d %b %Y %H:%M:%S +0000')
        self.writer.add_document(
            path=filename,
            content=text,
            title=title,
            subject=subject,
            description=description,
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


