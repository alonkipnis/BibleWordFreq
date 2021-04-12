# pipeline: data engineering

import pandas as pd
import numpy as np
import re
from nltk import everygrams
import logging

from biblical_scripts.pipelines.data_engineering.TextProcessing import TextProcessing
from biblical_scripts.extras.Convert import Convert

def _get_topics(topics_data) :
        """ Read topic from an external list and add as 
        a seperate field. 
        """

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
        
        return topic_verse
    
def _n_most_frequent_by_author(ds, n) :
    return ds.groupby(['author', 'feature'])\
            .count()\
            .sort_values(by='chapter', ascending=False)\
            .reset_index()\
            .groupby(['author'])\
            .head(n).filter(['feature'])

def _n_most_frequent(ds, n) :
    return ds.groupby(['feature'])\
            .count()\
            .sort_values(by='chapter', ascending=False)\
            .reset_index()\
            .head(n).filter(['feature'])

def build_vocab(data, params, known_authors) :
    n = params['no_tokens']
    by_author = params['by_author']
    
    ds = data[data.author.isin(known_authors)]
    ds = ds[~ds.feature.str.contains(r"\[[a-zA-Z0-9]+\]")] # remove code
    if by_author :
        r = _n_most_frequent_by_author(ds, n)
    else :
        r = _n_most_frequent(ds, n)
    #import pdb; pdb.set_trace()
    r = r.drop_duplicates()
    
    logging.info(f"Obtained a vocabulary of {len(r)} features")
    return r
    
def add_topics(data, topics_data) :
    """
    Add topic information to author-doc-term data frame
    """
    def range_verse(r) :
        if not pd.isna(r) :
            ls = r.split('-')
            return list(range(int(ls[0]), int(ls[1])+1))

    topics_data.loc[:, 'num'] = topics_data.verses.apply(range_verse)
    topics_data = topics_data.explode('num')

    # rename 'verse' with book and chapter info
    topics_data.loc[:,'verse'] = topics_data.book.astype(str) + "." + topics_data.chapter.astype(str)\
     + '.' + topics_data.num.astype(str)

    #get topic by verse
    topic_verse = topics_data.filter(['topic', 'verse']).set_index('verse')
    
    data = data.set_index('verse').join(topic_verse)  #merge into original dataset
    return data

def process_data(data : pd.DataFrame, params) -> pd.DataFrame :
    """
    Issue with 3-parts tokens  xxx/xxx/xxx (resolved?)
    """
    tp = TextProcessing(**params)
    data_proc = tp.proc(data)
    return data_proc

def add_convert(data : pd.DataFrame, data_org) -> pd.DataFrame :
    """
    Add a column showing converted features 
    """
    convert=Convert(data_org)
    try :
        data['feature-trans'] = data['feature'].apply(str).apply(eval).apply(convert.to_term)
    except :
        data['feature-trans'] = data['feature'].apply(str).apply(convert.to_term)
    return data