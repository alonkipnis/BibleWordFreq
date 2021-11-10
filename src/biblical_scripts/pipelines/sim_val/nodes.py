# pipeline: data science val
# project: bib-scripts

import pandas as pd
import numpy as np
import logging

from typing import Dict, List
from biblical_scripts.pipelines.sim.nodes import (build_model, model_predict, _prepare_data)
from sklearn.model_selection import KFold

#import warnings
#warnings.filterwarnings("error")

def _val_pipeline(data_train : pd.DataFrame, data_test : pd.DataFrame, 
                 vocabulary, model_params) -> float :
    """
    Validation pipeline: 
    1. model construction using training data
    2. prediction of testing data
    """
    md, _ = build_model(data_train, vocabulary, model_params)
    labels = data_test[['doc_id', 'author']].drop_duplicates()
    data_test.loc[:,'author'] = 'UNK' # obscure true labels
    df1 = model_predict(data_test, md)
    df1 = df1.drop('author', axis=1).merge(labels, on='doc_id', how='left')
    #import pdb; pdb.set_trace()
    return df1

def cross_validation(data, vocabulary, model_params) :
    """
    Evaluate doc-corpus similarities of docuements in a 
    cross validation setting. If n_fold is not provided 
    use leave-one-out (n-fold where n is the number of documents) 
    
    Args:
    -----
    data          entire dataset
    vocabulary    model vocabulary 
    model_params  instructions for model construction
    n_fold        number of CV sets (defult = num_of_docs)  
    
    """
    
    ds = _prepare_data(data)
    lo_docs = ds.doc_id.unique()
    n_known = len(lo_docs)

    n_fold = model_params.get('n_fold', n_known)  # defult is leave-one-out

    kf = KFold(n_splits=n_fold, shuffle=True)
    
    res = pd.DataFrame()
    for train_index, test_index in kf.split(lo_docs):
        docs_train, docs_test = lo_docs[train_index], lo_docs[test_index]
        rec = _val_pipeline(ds[ds.doc_id.isin(docs_train)],
                           ds[ds.doc_id.isin(docs_test)],
                           vocabulary, model_params)
        res = res.append(rec, ignore_index=True)
        #import pdb; pdb.set_trace()
    
    res['n_fold'] = n_fold
    return res

def bagging(data, vocabulary, model_params, bagging_params) :
    """
    # perform bagging several times
    # each time create a random train test corpora, with proportion 0.75, 0.25
    # then sample the test 0.8, 0.2, and calculate accuracy score

    Evaluate doc-corpus similarities of docuements of
    known authorship in a leave-one-out fashion 
    (i.e., n-fold cross validation where n is the number 
    of documents)
    
    Args:
    -----
    data         entire dataset
    vocabulary   model vocabulary 
    """
    lo_authors = data.author.unique().tolist()
    for itr in range(bagging_params['nBS']) :
        for grp in data.groupby('author') :
            # >>>HERE!!
            # ask Shira why do we need bagging. 
            # usually bagging is for ensemble methods, but here
            # it looks like we want to use it to evalaute robustness ?!

            bs_ds_train = grp['doc_id'].sample(frac = .8)
            bs_ds_test = grp['doc_id'][grp['doc_id'] != bs_ds_train]

            res1 = cross_validation(data_bs, vocabulary, params_model)
            res1['itr_BS'] = itr
            res = res.append(res1, ignore_index=True)
    return res

