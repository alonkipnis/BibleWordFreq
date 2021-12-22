# pipeline: data science
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from biblical_scripts.extras.AuthAttLib.MultiDoc import CompareDocs
from typing import Dict, List

pd.options.mode.chained_assignment = None


def _build_model(data, vocab, model_params):
    """
    Build author/corpus comparison model

    The model is essentially word-frequency table for each author.
    It supports the evaluation of binomial allocation P-values and
    the computation of Fisher test statistics, HC test statistics,
    and other measures.

    Params:
        :data:  is a dataframe with columns feature, class, doc_id
        :vocab: is a list of features to consider
        :model_params: is a dictionary containing parameters for CompareDocs model

    """
    md = CompareDocs(vocabulary=vocab, **model_params)
    ds = _prepare_data(data)
    logging.info(f"Building a model using {len(ds.doc_id.unique())} documents. ")
    train_data = {}
    lo_auth = ds.author.unique()
    for auth in lo_auth:
        train_data[auth] = ds[ds.author == auth]

    md.fit(train_data)
    return md


def reduce_vocab(data: pd.DataFrame,
                 vocabulary: pd.DataFrame,
                 model_params: Dict) -> pd.DataFrame:
    """
    Returns a reduced version of the original vocabulary with
    possible based on model_params['feat_reduction_method']

    Args:
        data            data used for building the model
        vocabulary      large vocabulary
        model_params    configurations for model construction
                        and feature selection.
    Returns:
        the new vocabulary

    """

    reduction_method = model_params['feat_reduction_method']

    if reduction_method == "none":
        return vocabulary

    vocab = vocabulary.feature.astype(str).to_list()
    md = _build_model(data, vocab, model_params)

    if reduction_method == "div_persuit":
        df_res = md.HCT()
        r = df_res[df_res.thresh].reset_index()
    if reduction_method == "one_vs_many":
        r = md.HCT_vs_many_filtered().reset_index()

    logging.info(f"Reducing vocabulary to {len(r.feature)} features")
    return r


def _prepare_data(data):
    """
    Arrange data in a way suitable for inference

    params:
        :data:          The dataset in author-chapter-feature format
    """

    ds = data.copy()
    if 'doc_id' in ds.columns:
        return ds
    else:
        ds = ds.rename(columns={'chapter': 'doc_id'}).dropna()
        ds = ds.filter(['author', 'feature', 'token_id', 'doc_id', 'to_report'])
        ds['doc_tested'] = ds['doc_id']
        ds['doc_id'] = ds['author'] + '|' + ds['doc_id'].astype(str)  # this is to make
        # sure doc_id is unique, as sometimes there are multiple authors per chapter
        ds['len'] = ds.groupby('doc_id').feature.transform('count')
    return ds


def build_model(data: pd.DataFrame,
                vocabulary: pd.DataFrame, model_params) -> CompareDocs:
    """
    Build authorship analysis model. Reduces vocabulary if needed
    
    Args:
        data        DataFrame with columns: 'doc_id', 'author', 'term'
        vocabulary  DataFrame with column 'feature'

    Return:
        CompareDocs model
    """

    df_vocabulary = reduce_vocab(data, vocabulary, model_params)
    vocab = df_vocabulary.feature.astype(str).to_list()
    return _build_model(data, vocab, model_params), df_vocabulary


def filter_by_author(df: pd.DataFrame, lo_authors=[],
                     lo_authors_to_merge=[]) -> pd.DataFrame:
    """
    Removes whatever author is not in lo_authors. 
    
    Adds chapter info for whatever author in 
    lo_authors_to_merge so that all chapters by these
    authors are considered as one document
    """
    #df = df[df.to_report] # uncomment here if you only want to use
                          # original 50 chapters


    if lo_authors_to_merge:
        idcs = df.author.isin(lo_authors_to_merge)
        df.loc[idcs, 'chapter'] = 'chapter0'

    if lo_authors:
        return df[df.author.isin(lo_authors)]
    else:
        return df


def model_predict(test_data: pd.DataFrame, model) -> pd.DataFrame:
    """
    Args:
        :data:  a dataframe representing tokens by docs by corpus
        :model: an instance of CompareDocs
    
    Returns:
    :df_res: Each row is the comparison of a doc against a corpus
    """

    ds = _prepare_data(test_data)

    observable = r"|".join(model.measures)  # r"HC|Fisher|chisq"
    df_res = pd.DataFrame()
    for doc_id in ds.doc_id.unique():
        doc_to_test = ds[ds.doc_id == doc_id]
        auth = doc_to_test.author.values[0]
        df_rec = model.test_doc(doc_to_test, of_cls=auth)

        r = df_rec.iloc[:, df_rec.columns.str.contains(observable)].mean()
        r['doc_id'] = doc_id
        r['author'] = auth
        r['len'] = len(doc_to_test)
        df_res = df_res.append(r, ignore_index=True)

    df_eval = df_res.melt(['author', 'doc_id', 'len'])
    df_eval['doc_tested'] = df_eval['doc_id']  # for compatibility with sim_full
    return df_eval


def evaluate_accuracy(df: pd.DataFrame,
                      report_params, parameters) -> pd.DataFrame:
    def _eval_succ(df):
        df['wrt_author'] = df['variable'].str.extract(r'([^:]+):')
        idx_min = df.groupby(['doc_id', 'author'])['value'].idxmin()
        res_min = df.loc[idx_min, :].rename(columns={'wrt_author': 'most_sim'})
        res_min.loc[:, 'succ'] = res_min.author == res_min.most_sim
        return res_min

    value = report_params['value']
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]

    res = _eval_succ(df1)
    res['param'] = str(parameters)
    return res
