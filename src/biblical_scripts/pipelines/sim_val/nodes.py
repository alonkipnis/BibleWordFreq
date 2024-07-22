# pipeline: data science val
# project: bib-scripts

import pandas as pd
import numpy as np
import logging

from typing import Dict, List
from biblical_scripts.pipelines.sim.nodes import (
    build_model, model_predict, _prepare_data)
from sklearn.model_selection import KFold
from biblical_scripts.pipelines.data_engineering.nodes import build_vocab


def _val_pipeline(data_train: pd.DataFrame, data_test: pd.DataFrame,
                  vocabulary, model_params) -> float:
    """
    Validation pipeline: 
    1. model construction using training data
    2. prediction of testing data
    """
    md, _ = build_model(data_train, vocabulary, model_params)
    labels = data_test[['doc_id', 'author']].drop_duplicates()
    data_test.loc[:, 'author'] = 'UNK'  # obscure true labels
    df1 = model_predict(data_test, md)
    df1 = df1.drop('author', axis=1).merge(labels, on='doc_id', how='left')
    return df1


def cross_validation(data, params_vocab, params_model):
    """
    Evaluate doc-corpus similarities of documents in a
    cross validation setting. If n_fold is not provided 
    use leave-one-out (n-fold where n is the number of documents) 
    
    Args:
    -----
    data          entire dataset
    vocabulary    model vocabulary 
    model_params  instructions for model construction
    n_fold        number of CV sets (default = num_of_docs)
    
    """

    ds = _prepare_data(data)
    lo_docs = ds.doc_id.unique()
    n_known = len(lo_docs)

    n_fold = params_model.get('n_fold', n_known)  # defult is leave-one-out
    if n_fold < 0:
        n_fold = n_known

    kf = KFold(n_splits=n_fold, shuffle=True)

    res = pd.DataFrame()
    for train_index, test_index in kf.split(lo_docs):
        docs_train, docs_test = lo_docs[train_index], lo_docs[test_index]
        data_train = ds[ds.doc_id.isin(docs_train)]
        #
        vocabulary = build_vocab(data_train, params_vocab)
        print(f"Docs in test set: {docs_test}")
        rec = _val_pipeline(data_train,
                            ds[ds.doc_id.isin(docs_test)],
                            vocabulary, params_model)
        res = res.append(rec, ignore_index=True)

    res['n_fold'] = n_fold
    return res
