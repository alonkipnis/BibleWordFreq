# pipeline: data science 1
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from biblical_scripts.pipelines.data_science.AuthorshipAttribution.AuthAttLib import AuthorshipAttributionDTM
from biblical_scripts.pipelines.data_science.AuthorshipAttribution.MultiDoc import CompareDocs
from typing import Dict, List


def build_reduced_vocab(data, vocabulary, reduction_method, model_params) :
    if reduction_method == "none" :
        return vocabulary
    md = build_model(data, vocabulary, model_params)
    if reduction_method == "div_persuit" :
        df_res = md.HCT()
        r = df_res[df_res.thresh].reset_index()
    if reduction_method == "one_vs_many" :
        r = md.HCT_vs_many_filtered().reset_index()
    logging.info(f"Reduced vocabulary to {len(r.feature)} features")
    return r

def _prepare_data(data) :
    if 'doc_id' in data.columns :
        return data
    else :
        ds = data.rename(columns = {'chapter' : 'doc_id'}).dropna()
        ds = ds[['author', 'feature', 'token_id', 'doc_id']]
        ds['doc_id'] += ' by '
        ds['doc_id'] += ds['author'] #sometimes there are multiple authors per chapter
    return ds

def build_model(data : pd.DataFrame, vocabulary : pd.DataFrame, model_params) -> CompareDocs :
    """
    Returns a model object
    
    Args:
    -----
    data    doc_id|author|term
    
    TODO: can implement vocab reduction as part of the model
    """
    
    vocab = vocabulary.feature.astype(str).to_list()
    md = CompareDocs(vocabulary=vocab, **model_params)
    
    ds = _prepare_data(data)
    train_data = {}
    lo_auth = ds.author.unique()
    for auth in lo_auth :
        train_data[auth] = ds[ds.author==auth]

    md.fit(train_data)
    return md
    
def filter_by_author(df, lo_authors) :
    return df[df.author.isin(lo_authors)]
    
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
    return df_eval

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

def OLD_evaluate_accuracy(df_res: pd.DataFrame, known_authors : List,
                          min_length_to_report : int, params) -> pd.DataFrame :
    """
    Evaluate accuracy of attribution in min discrepancy manner
    
    Args:
    min_length_to_report    only try to attribute docs of length larger than that
    known_authors           only compare against corpora from this list
    
    Returns:
    average accuracy for each similarity measure
    """
    def prob_of_succ(df, value, known_authors, min_length=min_length_to_report, plot=False) :
        df = df[df.len >= min_length]
        df.loc[:,value] = df[value].astype(float)
        res1 = df[df.author.isin(known_authors) & df.wrt_author.isin(known_authors)].reset_index()
        idx_min = res1.groupby(['doc_id', 'author'])[value].idxmin()
        res_min = res1.loc[idx_min, :]
        res_min.loc[:, 'succ'] = res_min.author == res_min.wrt_author

        succ_per_doc = res_min.groupby('doc_id').succ.mean()
        return succ_per_doc.mean()

    res = {}
    for v in ['HC', 'HC_rank', 'chisq', 'chisq_rank', 'log-likelihood','log-likelihood_rank', 'Fisher']  :
        res[v] = prob_of_succ(df_res, value = v, known_authors=known_authors)
        
    res['params'] = str(params)
    return pd.DataFrame(res, index=[0])

def OLD_compute_sim(data : pd.DataFrame, vocabulary : pd.DataFrame,
                    params, known_authors : List) -> pd.DataFrame :
    """
    Build model for comparing docs and authors based on word-frequencies
    
    Args:
    data        a dataframe representing tokens by docs by corpus
    vocabulary  a list of words by which docs are compared
    known_authors   only compare against corpora from this list
    
    Returns:
    df_res      Each row is the comparison of a doc against a corpus in known_authors
    """
    
    ds = data.rename(columns = {'feature' : 'term', 'chapter' : 'doc_id'}).dropna()
    ds['len'] = ds.groupby('doc_id').transform(pd.Series.count).iloc[:,0]
    par = {}
    ds = data.rename(columns = {'feature' : 'term', 'chapter' : 'doc_id'}).dropna()
    model = AuthorshipAttributionDTM(ds, **params, vocab=list(vocabulary.feature.values))
    
    # adding doc-length info
    doc_lengths = pd.DataFrame(ds.groupby('doc_id').count().rename(columns={'term' : 'len'}).len)
    df_res = model.compute_inter_similarity(LOO = True, wrt_authors=known_authors)
    df_res = df_res.merge(doc_lengths, on = 'doc_id')
    return df_res
