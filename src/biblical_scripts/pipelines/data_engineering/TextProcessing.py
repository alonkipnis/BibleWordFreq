import pandas as pd
import numpy as np
import re
from nltk import everygrams
import logging


def _extract_ngrams(df, key, by=[], ng_range=(1,1), pad=False) :
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


def extract_prefix_suffix(data) :
    """
    Returns a dataframe with column 'feature' and as many rows as 
    number of prefix + lemmas + suffixes

    """
    suff = data[data.morph.str.contains(r'/S[dhnp][1-3][bcfm][dps]')]

    data['lemma'] = data.lemma.str.extract(r'([\/a-z0-9]+)')
    data['lemma_split'] = data.lemma.str.split(r"/| ")
    data_lem_ex = data.explode('lemma_split')
    pref = data_lem_ex[data_lem_ex.lemma_split.isin(
        ["a","b","c", "d", "e", "l", "m", "k", "i", "s"])]
    pref = pref.drop('lemma_split', axis=1)

    #pref = data[data.morph.str.contains(r'^[HA][A-Z][a-z]?/[^S]')]

    data.loc[:,'feature'] = data.lemma.str.extract(
                                r'(?:^[a-z]/)?([A-Za-z0-9]+)', expand=False)
    data.loc[:,'morph'] = data.morph.str.extract(
                        r'(?:[HA][A-Z][a-z]?/)?([A-Za-z0-9]+)', expand=False)
    data.loc[:,'POW'] = 'main'

    suff.loc[:, 'feature'] = '[' + suff.morph.str.extract(
                        r'(S[dhnp][1-3][bcfm][dps])', expand=False) + ']'

    suff.loc[:, 'morph'] = suff.morph.str.extract(r'(S[dhnp][1-3][bcfm][dps])',
                                                  expand=False)
    suff.loc[:, 'POW'] = 'suffix'

    pref.loc[:, 'feature'] = pref.lemma.str.extract(r'(^[a-z]|l)/?',
                                                     expand=False)
    pref.loc[:, 'morph'] = pref.morph.str.extract(r'([HA][A-Z][a-z]?)/',
                                                     expand=False)
    pref.loc[:, 'POW'] = 'prefix'
    return pd.concat([pref, data, suff]).sort_values(by="token_id")


class TextProcessing :
    """
        1) removes or replaces specific lemmas based on 
        morphological codes
        2) Arranges features as lemma n-grams
        3) Usually initiallized with certain global 
        parameters (e.g. conf/base/params.yml)

        Adds brackets around lemmas to indicate special 
        instruction for counting lemmas in subsequent parts:
        [lemma] = ignore lemma
        <lemma> = count lemma as a sub-group based on re

    """

    def __init__(self, **kwargs) :
        self.to_remove = kwargs.get('to_remove',[])
        self.to_replace = kwargs.get('to_replace',[])
        self.extract_prefix = kwargs.get('extract_prefix', False)
        self.count_suffix = kwargs.get('extract_suffix', False)
        ng_max = kwargs.get('ng_max', 1)
        ng_min = kwargs.get('ng_min', 1)
        self.ng_range = (ng_min, ng_max)
        self.pad = kwargs.get('pad', False)  # both sides
        
    def _proc(self, raw_data) :

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
            data['feature'].replace({r"\[(S[dhnp][1-3][bcfm][dps])\]" : r"\1"},
                                     inplace=True, regex=True)

        data.loc[:,'feature (org)'] = data['feature']

        for cd in self.to_remove :
            logging.info(f"Removing {cd}")
            data.loc[data.morph.str.contains(fr'(?:^|[H/])({cd})'),'feature'] = f"[{cd}]"

        for cd in self.to_replace :
            logging.info(f"Replacing {cd}")
            data.loc[data.morph.str.contains(fr'(?:^|[H/])({cd})'),'feature'] = f"<{cd}>"

        data['token_id'] = data.index

        self._data = data.filter(['token_id', 'author', 'chapter', 'verse', 
                    'feature', 'feature (org)', 'morph', 'term', 'lemma', 
                    'POW'])
        return self._data

    def _proc_ng(self, raw_data) :
        # upadate self.list_of_trans
        data = self._proc(raw_data)
        data_ng = _extract_ngrams(data, ng_range=self.ng_range, key='feature',
            by=['author', 'chapter', 'verse'], pad=self.pad)

        # to track what tokens are grouped where:
        dfmap = _extract_ngrams(data, ng_range=self.ng_range, key='token_id',
            by=['author', 'chapter', 'verse'], pad=self.pad)

        data_ng['token_id'] = dfmap.token_id

        exf = data_ng['feature'].explode()
        self._data_ng = data_ng
        return data_ng
    
    def proc(self, raw_data) :
        if self.ng_range == (1,1) :
            return self._proc(raw_data)
        else :
            return self._proc_ng(raw_data)


    def inv_trans(self, tid) :
        return self._data[self._data.token_id == tid]

    def inv_trans_ng(self, tid) :
        dfng = self._data_ng
        return dfng[dfng.token_id.apply(lambda x: tid in x)]
