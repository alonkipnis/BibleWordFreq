"""
Module for reading from OSHB. 
OpenBible project inforation: https://hb.openscriptures.org/index.html
Data folder is avaialbe at https://github.com/openscriptures/morphhb/tree/master/wlc
Information on morphological code is available at https://hb.openscriptures.org/parsing/HebrewMorphologyCodes.html

Usage:

Create OSHBData object by providing the path to the local data folder and 
the catalog file containing information on what parts to read. 

"""

from xml.dom.minidom import parse, parseString
import pandas as pd
import logging
from nltk import everygrams
import numpy as np
import re


def extract_ngrams(df, key, by=[], ng_range=(1,1), pad=False) :
    """
        nest terms as ngrams 
    Args:
    -----
    df : DataFrame with columns: term, author, doc_id
    ng_range : (min_gram, max_gram) 
    by : list containing fileds to group by
    pad : whether to add <start>/<end> symbols when extracting n-grams
    """

    if pad :
        new_df = df.groupby(by)[key]\
                .apply(lambda x : list(everygrams(x, min_len=ng_range[0], 
                                              max_len=ng_range[1], 
                                             pad_left=True,
                                             pad_right=True,
                                             left_pad_symbol='<start>',
                                             right_pad_symbol='<end>'
                                             )))\
        .explode()\
        .reset_index()
    else :
        new_df = df.groupby(by)[key]\
            .apply(lambda x : list(everygrams(x, min_len=ng_range[0], 
                                              max_len=ng_range[1]
                                             )))\
        .explode()\
        .reset_index()
    return new_df

def remove_POS(df, code) : 
    df.loc[df.morph.str.contains(fr'(?:^|[H/])({code})'), 'feature'] = f"[{code}]"
    return df
    
def replace_POS(df, code) :
    df.loc[df.morph.str.contains(fr'(?:^|[H/])({code})'), 'feature'] = f"<{code}>"
    return df

def extract_prefix_suffix(data) :
    """
    Returns a dataframe with column 'feature' and as many rows as 
    number of prefix + lemmas + suffixes

    """
    suff = data[data.morph.str.contains(r'/S[dhnp][1-3][bcfm][dps]')]
    pref = data[data.morph.str.contains(r'^[HA][A-Z][a-z]?/[^S]')]

    data.loc[:,'feature'] = data.lemma.str.extract(r'(?:^[a-z]/)?([A-Za-z0-9]+)', expand=False)
    data.loc[:,'morph'] = data.morph.str.extract(r'(?:[HA][A-Z][a-z]?/)?([A-Za-z0-9]+)', expand=False)
    data.loc[:,'POW'] = 'main'

    suff.loc[:, 'feature'] = '[' + suff.morph.str.extract(r'(S[dhnp][1-3][bcfm][dps])', expand=False) + ']'
    suff.loc[:, 'morph'] = suff.morph.str.extract(r'(S[dhnp][1-3][bcfm][dps])', expand=False)
    suff.loc[:, 'POW'] = 'suffix'

    pref.loc[:, 'feature'] = pref.lemma.str.extract(r'(^[a-z]|l)/?', expand=False)
    pref.loc[:, 'morph'] = pref.morph.str.extract(r'([HA][A-Z][a-z]?)/', expand=False)
    pref.loc[:, 'POW'] = 'prefix'
    return pd.concat([pref, data, suff]).sort_values(by="token_id")

def read_catalog(catalog_file) :
    """
    The catalog file is a csv file with columns: 'author', 'book', 'chapter', 'verses'
    """
    list_of_ref = pd.read_csv(catalog_file)
    list_of_ref.loc[:,'verses'] = list_of_ref.verses.astype(str)\
                            .apply(lambda vs : vs.strip(";").split(";"))
    list_of_ref = list_of_ref.explode('verses')
    list_of_ref.verses = list_of_ref.verses.replace('nan','all')
    return list_of_ref
    
def read_from_morph(path, catalog) : 

    def read_chapter(book, chapter, verse_set = []) :
        df = pd.DataFrame()
        bookxml = parse(path + '/' +book + '.xml')
        chapterlist = bookxml.getElementsByTagName('chapter')
        chapterlist = [ch for ch in chapterlist if ch.attributes['osisID'].value == book+'.'+str(chapter)]
        for chap in chapterlist:
            verselist = chap.getElementsByTagName('verse')
            for verse in verselist:
                mywelements = verse.getElementsByTagName('w')
                for el in mywelements:
                    vrs = verse.attributes['osisID'].value
                    vrs_numeric = int(vrs.split('.')[-1])
                    if len(verse_set) == 0 or vrs_numeric in verse_set :
                        df = df.append({'lemma' : el.attributes['lemma'].value,
                            'morph' : el.attributes['morph'].value,
                            'term' : el.firstChild.data,
                            'chapter' : chap.attributes['osisID'].value,
                            'verse' : vrs
                            }, ignore_index=True)
        return df

    data = pd.DataFrame()
    for c in catalog.groupby(['author', 'book', 'chapter']) :
        for i,r in enumerate(c[1].verses) :
            if r == 'all' :
                vs = []
            else :
                try :
                    a, b = r.split('-')
                except ValueError:
                    a = b = r
                vs = list(range(int(a),int(b)+1))
            logging.debug(f"Reading: author={c[0][0]}, book={c[0][1]}, chapter={c[0][2]}, verse_set={vs}")
            df = read_chapter(book = c[0][1], chapter = c[0][2], verse_set = vs)
            df.loc[:,'author'] = c[0][0]
            data = data.append(df, ignore_index = True)
    return data

def read_hash(hash_file) :
    try :
        with open(hash_file,'r') as fl:
            old_hash = fl.read()
        return int(old_hash)
    except :
        logging.debug(f"Cannot find hash file {hash_file}. Returning hash_val = '-1' ")
        logging.debug(f"Cannot find hash file {hash_file}")
        return "-1"
    
def store_hash(hash_val, hash_file) :
    with open(hash_file, 'w') as fl:
        fl.write(str(hash_val))
    
class OSHBData :
    def __init__(self, raw_data_path, catalog_file,
                 saved_data_path='./BiblicalScript_data.csv',
                  hash_file='./OSHB.hash',
                  force_saved=False) :
        
        self.catalog_file = catalog_file
        self.raw_data_path = raw_data_path
        
        logging.info(f"Reading Catalog File: {catalog_file}...")
        self.catalog = read_catalog(self.catalog_file)
        logging.info(f"Found {len(self.catalog)} entries in catalog.")
        
        old_hash = read_hash(hash_file)
        curr_hash = hash(str(self.catalog.verses.values))
        
        # indicate whether catalog file has changed since last read
        self._has_changed = old_hash != curr_hash
        
        if self._has_changed and not force_saved:
            logging.info(f"Catalog has changed since last reading. Extracting from raw data based on {catalog_file}")
            self._data = read_from_morph(self.raw_data_path, self.catalog)
            logging.info(f"Saving extracted data to {saved_data_path} for future use...")
            self._data.to_csv(saved_data_path)
            logging.info(f"Storing catalog hash value in {hash_file}.")
            store_hash(curr_hash, hash_file)

        else :
            logging.info(f"Reading from {saved_data_path}...")
            self._data = pd.read_csv(saved_data_path)
    
        self.dictionary1 = dict([(k,v) for k,v in zip(self._data.lemma.values, self._data.morph.values)])
    
        self.dictionary = dict([(k,v) for k,v in zip(self._data.lemma.values, self._data.term.values)]
                  + [('c','ו'),('d','ה'), ('b','ב'), ('l', 'ל'), ('k', 'כ')])


    def lemma_to_term(self, lemma) :
        """
        Convert lemma code to a Hebrew word. If can't 
        find a matching word, return lemma code.
        """
        dictionary = self.dictionary
        return dictionary.get(lemma, 
           dictionary.get(lemma + ' a',
           dictionary.get(lemma + ' b',
           dictionary.get(lemma + ' c',
           dictionary.get(lemma + ' d',
           dictionary.get(lemma + ' e',
           dictionary.get(lemma + ' l',
           dictionary.get(lemma + ' m',
           dictionary.get('a/'+lemma,
           dictionary.get('b/'+lemma,
           dictionary.get('c/'+lemma,
           dictionary.get('c/'+lemma + ' b',
           dictionary.get('d/'+lemma,
           dictionary.get('e/'+lemma,
           dictionary.get('k/'+lemma,
           dictionary.get('l/'+lemma,
           dictionary.get('m/'+lemma,
           dictionary.get('m/'+lemma,
                    lemma))))))))))))))))))
    
    def lemma_to_morph(self, lemma) :
        return self.dictionary1[lemma]
        
    def ngram_to_term(ng) :
        return [self.lemma_to_term(l) for l in ng]
    
    def get_data() :
        return self._data
            
    def get_topics(self, topics_file) :
        """ Read topic from an external list and add as 
        a seperate field. 
        """

        topics_data = pd.read_csv(topics_file)
        topics_data = topics_data.dropna()

        def range_verse(r) :
            ls = r.split('-')
            return list(range(int(ls[0]), int(ls[1])+1))

        topics_data.loc[:, 'num'] = topics_data.verses.transform(range_verse)
        topics_data = topics_data.explode('num')

        # rename 'verse' with book and chapter info
        topics_data.loc[:,'verse'] = topics_data.book.astype(str) + "." + topics_data.chapter.astype(str)\
         + '.' + topics_data.num.astype(str)

        #get topic by verse
        topic_verse = topics_data.filter(['topic', 'verse']).set_index('verse')

        #merge into original dataset
        return self._data.set_index('verse').join(topic_verse)

class ProcessText :
    def __init__(self, **kwargs) :
        self.to_remove = kwargs.get('to_remove',[])
        self.to_replace = kwargs.get('to_replace',[])
        self.extract_prefix = kwargs.get('extract_prefix', False)
        self.count_suffix = kwargs.get('extract_suffix', False)
        self.ng_range = kwargs.get('ng_range', (1,1))
        self.pad = kwargs.get('pad', False)  # both sides

    def proc(self, raw_data) :
        """
        [lemma] is code for ignore 
        <lemma> is code for count as a sub-group

        """

        # upadate self.list_of_trans
        data = raw_data.copy()
        data.loc[:, 'feature'] = data['lemma']
    # remove/replace/extract specific codes
        data['token_id'] = data.index

        if self.extract_prefix :
            logging.info("Extracting prefixes and suffixes")
            data = extract_prefix_suffix(data)

        if self.count_suffix :
            logging.info("Counting suffixes")
            data['feature'].replace({r"\[(S[dhnp][1-3][bcfm][dps])\]" : r"\1"}, inplace=True, regex=True)

        data.loc[:,'feature (org)'] = data['feature']

        for cd in self.to_remove :
            logging.info(f"Removing {cd}")
            data.loc[data.morph.str.contains(fr'(?:^|[H/])({cd})'),
             'feature'] = f"[{cd}]"

        for cd in self.to_replace :
            logging.info(f"Replacing {cd}")
            data.loc[data.morph.str.contains(fr'(?:^|[H/])({cd})'),
             'feature'] = f"<{cd}>"

        data['token_id'] = data.index

        self._data = data.filter(['token_id', 'author', 'chapter', 'verse', 
                    'feature', 'feature (org)', 'morph', 'term', 'lemma', 
                    'POW'])
        return self._data

    def proc_ng(self, raw_data) :
        # upadate self.list_of_trans
        data = self.proc(raw_data)
        #data = data[~data.feature.str.match('^\[.+\]$')] # square brackets
        # indicate 'ignore'
        data_ng = extract_ngrams(data, ng_range=self.ng_range, key='feature',
            by=['author', 'chapter', 'verse'], pad=self.pad)

        # to track what tokens are grouped where:
        dfmap = extract_ngrams(data, ng_range=self.ng_range, key='token_id',
            by=['author', 'chapter', 'verse'], pad=self.pad)

        data_ng['token_id'] = dfmap.token_id

        exf = data_ng['feature'].explode()
        # remove those marked with [ ]
        #data_ng = data_ng.loc[data_ng.feature.apply(
        #    lambda x : not (re.search(r"^\[.+\]$",x[0]) or re.search(r"^\[.+\]$",x[1]))),:]
        self._data_ng = data_ng
        return data_ng

    def inv_trans(self, tid) :
        return self._data[self._data.token_id == tid]

    def inv_trans_ng(self, tid) :
        dfng = self._data_ng
        return dfng[dfng.token_id.apply(lambda x: tid in x)]
    
        # use self.list_of_trans to inverse token



