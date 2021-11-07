# pipeline: data science
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from biblical_scripts.pipelines.data_science.AuthorshipAttribution.MultiDoc import CompareDocs
from typing import Dict, List

def _build_model(data, vocab, model_params) :
    md = CompareDocs(vocabulary=vocab, **model_params)
    ds = _prepare_data(data)
    train_data = {}
    lo_auth = ds.author.unique()
    for auth in lo_auth :
        train_data[auth] = ds[ds.author==auth]
    
    md.fit(train_data)
    return md

def reduce_vocab(data, vocabulary, model_params) -> pd.DataFrame :
    """
    Build a model using original vocabulary with 
    possible reduction of vocabulary elements 
    based on model_params['feat_reduction_method']

    """
    
    reduction_method = model_params['feat_reduction_method']
    if reduction_method == "none" :
        return vocabulary
    
    vocab = vocabulary.feature.astype(str).to_list()
    md = _build_model(data, vocab, model_params)
    
    if reduction_method == "div_persuit" :
        df_res = md.HCT()
        r = df_res[df_res.thresh].reset_index()
    if reduction_method == "one_vs_many" :
        r = md.HCT_vs_many_filtered().reset_index()
    logging.info(f"Reducing vocabulary to {len(r.feature)} features")
    return r

def _prepare_data(data) :
    if 'doc_id' in data.columns :
        return data
    else :
        ds = data.rename(columns = {'chapter' : 'doc_id'}).dropna()
        ds = ds[['author', 'feature', 'token_id', 'doc_id']]
        ds['doc_tested'] = ds['doc_id']
        ds['doc_id'] += ' by '
        ds['doc_id'] += ds['author'] #sometimes there are multiple authors per chapter
        ds['len'] = ds.groupby('doc_id').feature.transform('count')
    return ds

def build_model(data : pd.DataFrame, vocabulary : pd.DataFrame, model_params) -> CompareDocs :
    """
    Returns a model object
    
    Args:
    -----
    data    doc_id|author|term
    
    TODO: can implement vocab reduction as part of the model
    """
    
    vocabulary = reduce_vocab(data, vocabulary, model_params)
    
    vocab = vocabulary.feature.astype(str).to_list()
    return _build_model(data, vocab, model_params), vocabulary
    
def filter_by_author(df : pd.DataFrame, lo_authors=[], lo_authors_to_merge=[]) -> pd.DataFrame :
    """
    Remove whatever author is not in lo_authors. Overwrite doc_id info
    for whatever author in lo_authors_to_merge. 
    """
    
    if lo_authors_to_merge :
        idcs = df.author.isin(lo_authors_to_merge)
        df[idcs]['chapter'] = 'chapter0'
    
    if lo_authors :
        return df[df.author.isin(lo_authors)]
    else :
        return df
    
def model_predict(data_test : pd.DataFrame, model) -> pd.DataFrame :
    """
    Args:
    data        a dataframe representing tokens by docs by corpus
    model       CompareDocs
    
    Returns:
    df_res      Each row is the comparison of a doc against a corpus in known_authors
    """
    
    ds = _prepare_data(data_test)
    
    observable = r"|".join(model.measures) #r"HC|Fisher|chisq"
    df_res = pd.DataFrame()
    for doc_id in ds.doc_id.unique() :
        tested_doc = ds[ds.doc_id==doc_id]
        auth = tested_doc.author.values[0]
        df_rec = model.test_doc(tested_doc, of_cls = auth)
        r = df_rec.iloc[:,df_rec.columns.str.contains(observable)].mean()
        r['doc_id'] = doc_id
        r['author'] = auth
        r['len'] = len(tested_doc)
        df_res = df_res.append(r, ignore_index=True)
    
    df_eval = df_res.melt(['author', 'doc_id', 'len'])
    df_eval['doc_tested'] = df_eval['doc_id'] # for compatibility with sim_full
    return df_eval

def _report_table(sim_res) :
    """
    Output table indicating accuracy of attribution
    """
    
    res_tbl = sim_res.pivot('corpus','doc_id','value')
    lo_corpora = sim_res.corpus.unique().tolist()
    cmin = res_tbl.idxmin().rename('min_corpus')
    res_tbl = res_tbl.append(cmin)
    res_tbl.loc['author', :] = [r[1] for r in res_tbl.columns.str.split(' by ')]
    res_tbl.loc['succ', :] = res_tbl.loc['min_corpus',:] == res_tbl.loc['author',:]
    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ'],:].mean(1)
    return res_tbl


def report_table_known(df, report_params) :
    value = report_params['value']
    known_authors = report_params['known_authors']
    
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:,'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1 = df1[df1.corpus.isin(known_authors)]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]
    df1 = df1[df1['author'].isin(known_authors)]  # bc its 'known_authors_only'
    
    return _report_table(df1)

def report_table_unknown(df, report_params) :
    
    value = report_params['value']
    known_authors = report_params['known_authors']
    
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:,'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1 = df1[df1.corpus.isin(known_authors)]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]
    df1 = df1[~df1['author'].isin(known_authors)] # bc its 'non known_authors_only'
    
    return _report_table(df1)

    
def evaluate_accuracy(df : pd.DataFrame, report_params, parameters) -> pd.DataFrame :
    
    def _eval_succ(df) :
        df['wrt_author'] = df['variable'].str.extract(r'([^:]+):')
        idx_min = df.groupby(['doc_id', 'author'])['value'].idxmin()
        res_min = df.loc[idx_min, :].rename(columns={'wrt_author' : 'most_sim'})
        res_min.loc[:, 'succ'] = res_min.author == res_min.most_sim
        return res_min

    value = report_params['value']
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]

    res = _eval_succ(df1)
    res['param'] = str(parameters)
    return res
