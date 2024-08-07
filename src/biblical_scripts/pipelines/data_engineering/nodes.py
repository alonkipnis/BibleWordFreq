# pipeline: data engineering
"""
This pipeline takes raw OSHB data (one token per row) and
applies several transformations based on 'preprocessing'
parameter in `parameters.yml`

"""

import pandas as pd
import logging
from typing import List

from biblical_scripts.pipelines.data_engineering.TextProcessing import TextProcessing
from biblical_scripts.extras.Convert import Convert


def _get_topics(topics_data):
    """ Read topic from an external list and add as
        a separate field.
        """

    def range_verse(r):
        ls = r.split('-')
        return list(range(int(ls[0]), int(ls[1]) + 1))

    topics_data.loc[:, 'num'] = topics_data.verses.transform(range_verse)
    topics_data = topics_data.explode('num')

    # rename 'verse' with book and chapter info
    topics_data.loc[:, 'verse'] = topics_data.book.astype(str) + "." + topics_data.chapter.astype(str) \
                                  + '.' + topics_data.num.astype(str)

    # get topic by verse
    topic_verse = topics_data.filter(['topic', 'verse']).set_index('verse')

    return topic_verse


def _n_most_frequent_by_author(ds, n):
    return ds.groupby(['author', 'feature']) \
        .count() \
        .sort_values(by='token_id', ascending=False) \
        .reset_index() \
        .groupby(['author']) \
        .head(n).filter(['feature'])

def _n_most_frequent_by_author(ds, n):
    return ds.groupby(['author', 'feature']) \
        .count() \
        .sort_values(by='token_id', ascending=False) \
        .reset_index() \
        .groupby(['author']) \
        .head(n).filter(['feature'])


def _n_most_frequent(ds, n):
    return ds.groupby(['feature']) \
        .count() \
        .sort_values(by='token_id', ascending=False) \
        .reset_index() \
        .head(n).filter(['feature'])


def merge_unknown(data: pd.DataFrame, unknown_authors: List) -> pd.DataFrame:
    """
    Mark chapters of authors in `unknown_authors` by their corpus name,
    so that such chapters are considered as a single documents.
    """
    idc = data.author.isin(unknown_authors)
    data.loc[idc, 'chapter'] = data.loc[idc, 'author']
    return data


def build_vocab(data, params):
    """
    Forms a vocabulary by counting all tokens, keeping only
    the 'no_tokens' most frequent ones.
    There are two modes:
    1) 'no_tokens' most frequent tokens across entire dataset
    2) 'no_tokens' most frequent tokens by each author
    (option 2 is preferred to avoid biases associated with the
    situation in which the author classes are unbalanced)

    Args:
        data    the dataframe containing one feature per row
        params:
          n         number of tokens to keep
          authors   which authors to consider
          by_author whether to use mode 1 or 2
    """

    n = params['no_tokens']
    by_author = params['by_author']
    authors = params['authors']

    ds = data[data.author.isin(authors)]
    ds = ds[~ds.feature.str.contains(r"\[[a-zA-Z0-9]+\]")]  # remove code
    if by_author:
        r = _n_most_frequent_by_author(ds, n)
    else:
        r = _n_most_frequent(ds, n)

    r = r.drop_duplicates()
    logging.info(f"Obtained a vocabulary of {len(r)} features")
    return r


def add_topics(data, topics_data):
    """
    Add topic information to author-doc-term data frame

    Topic information is taken from an extenral source. 
    
    This block is redundant as we currently do not use
    topic information.

    Issue with verse
    """

    def range_verse(r):
        if not pd.isna(r):
            ls = r.split('-')
            return list(range(int(ls[0]), int(ls[1]) + 1))

    topics_data.loc[:, 'num'] = topics_data.verses.apply(range_verse)
    topics_data = topics_data.explode('num')

    # rename 'verse' with book and chapter info
    topics_data.loc[:, 'verse'] = topics_data.book.astype(str) + "." + topics_data.chapter.astype(str) \
                                  + '.' + topics_data.num.astype(str)

    # get topic by verse
    topic_verse = topics_data.filter(['topic', 'verse']).set_index('verse')

    data = data.set_index('verse').join(topic_verse)  # merge into original dataset
    return data.reset_index()


def process_data(data: pd.DataFrame, params) -> pd.DataFrame:
    """
    Performs several transformation to features provided as
    lemmas and morphological information.

    See module TextProcessing in TextProcessing.py

    """

    tp = TextProcessing(**params)
    data_proc = tp.proc(data)
    return data_proc


def add_to_report(data: pd.DataFrame,
                  reference_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each chapter in the data, add a column indicating whether
    to include that chapter in the final report or not
    """

    df = reference_data
    df['chapter'] = df['book'] + "." + df['chapter'].astype((str))
    data = data.merge(df[['to_report', 'chapter', 'author']], how='inner', on=['author', 'chapter'])
    return data


def add_convert(data: pd.DataFrame, data_org: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column translating converted features back to terms as 
    much as possible (sometimes two different words has the same lemma)
    """
    convert = Convert(data_org)
    try:
        data['feature-trans'] = data['feature'].apply(str).apply(eval) \
            .apply(convert.to_term)
    except:
        data['feature-trans'] = data['feature'].apply(str).apply(convert.to_term)
    return data
