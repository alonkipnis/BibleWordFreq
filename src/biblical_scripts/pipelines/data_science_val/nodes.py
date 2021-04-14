# pipeline: data science val
# project: bib-scripts

import pandas as pd
import numpy as np
import logging

from typing import Dict, List
from biblical_scripts.pipelines.data_science.nodes import (build_reduced_vocab, build_model, evaluate_accuracy, model_predict, _prepare_data)
from sklearn.model_selection import KFold

#import warnings
#warnings.filterwarnings("error")

def _val_pipeline(data_train : pd.DataFrame, data_test : pd.DataFrame, 
                 vocabulary, model_params) -> float :
    vocab = build_reduced_vocab(data_train, vocabulary, 
                                model_params['feat_reduction_method'], model_params)
    md = build_model(data_train, vocab, model_params)
    labels = data_test[['doc_id', 'author']].drop_duplicates()
    data_test.loc[:,'author'] = 'UNK' # obscure true labels
    df1 = model_predict(data_test, md)
    df1 = df1.drop('author', axis=1).merge(labels, on='doc_id', how='left')
    #import pdb; pdb.set_trace()
    return df1

def cross_validation(data, vocabulary, model_params, report_params) :
    """
    Evaluate using cross validation 
    
    Args:
    -----
    data    entire dataset
    vocabulary   model vocabulary 
    
    """
    
    ds = _prepare_data(data)
    lo_docs = ds.doc_id.unique()
    n_known = len(lo_docs)
    n_fold = n_known
    kf = KFold(n_splits=n_fold, shuffle=True)
    
    res = pd.DataFrame()
    for train_index, test_index in kf.split(lo_docs):
        docs_train, docs_test = lo_docs[train_index], lo_docs[test_index]
        rec = _val_pipeline(ds[ds.doc_id.isin(docs_train)],
                           ds[ds.doc_id.isin(docs_test)],
                           vocabulary, model_params)
        res = res.append(rec, ignore_index=True)
        #import pdb; pdb.set_trace()
    acc = evaluate_accuracy(res, report_params, "cross_validation").succ.mean()
    logging.info(f"\n\n\tCV acc = {acc}\n\n")
    res['n_fold'] = n_fold
    return res
