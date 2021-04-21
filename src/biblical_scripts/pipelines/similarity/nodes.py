# pipeline: sim_full
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import logging

from typing import Dict, List
from biblical_scripts.pipelines.data_science.nodes import (build_model, model_predict, _prepare_data)

def _check_doc(ds, vocabulary, params_model) -> pd.DataFrame :
    """
    Build a model from training data not containing ds['author]=='TEST'
    and check ds['doc_id']=='TEST' against this model. Return the results of this test
    """
    
    data_train = ds[ds.author != 'TEST']
    md, _ = build_model(data_train, vocabulary, params_model)
    
    data_test = ds[ds.author == 'TEST']
    return model_predict(data_test, md)


def _test_doc(ds, vocabulary, params_model, params_sim, known_authors) :
    """
    Test the document marked as 'TEST' 
    """
    itr = 0
    res = pd.DataFrame()
    
    res1 = _check_doc(ds, vocabulary, params_model)  # evaluate wrt to corpus
    ds_doc = ds[ds.author=='TEST']
    res1['itr'] = itr
    res1['smp_len'] = len(ds_doc)
    res1['kind'] = 'org'
    res = res.append(res1, ignore_index=True) 

    tested_doc_id = ds_doc.doc_id.values[0]
    
    for ds1 in _gen_test_doc(ds, known_authors, params_sim) :
        itr += 1
        res1 = pd.DataFrame()
        try :
            res1 = _check_doc(ds1, vocabulary, params_model)
            res1 = res1[res1.variable.str.contains('-ext')]
        except : 
            import pdb; pdb.set_trace()
        res1['itr'] = itr
        res1['smp_len'] = len(ds1[ds1.author=='TEST'])
        res1['kind'] = 'ext'
        res = res.append(res1, ignore_index=True) 
    return res

def _gen_test_doc(ds0, known_authors, params) :
    """
    document-corpus pair generator
    
    Assumes that tested dcoument is marked by 'TEST' as author
    For each corpus, go over all existing doc_ids. 
    
    Future: include more sophisticated sampling methods
    
    """
        
    ds_doc = ds0[ds0.author == 'TEST']
    tested_doc_id = ds_doc.doc_id.values[0]
    
    for corp in known_authors :
        ds = ds0.copy()
        ds.loc[ds.author.isin([corp, 'TEST']), 'author'] = f'{corp}-ext'
        ds_pool = ds[(ds.author == f'{corp}-ext') & (ds.len >= params['min_length_to_consider'])]
    
        smp_pool = ds_pool.doc_id.unique()
        for smp in smp_pool :
            ds1 = ds.copy()
            ds1.loc[ds.doc_id == smp, 'author'] = 'TEST'
            yield ds1

            
def sim_full(data, vocabulary, params_model, params_sim, known_authors) :
    
    ds = _prepare_data(data)
    lo_docs = ds.doc_id.unique()
    
    res = pd.DataFrame()
    for doc in tqdm(lo_docs) :
        logging.info(f"Evaluating {doc}")
        ds1 = ds[ds.author.isin(known_authors) | (ds.doc_id == doc)]
        ds1.loc[ds1.doc_id == doc, 'author'] = 'TEST'
        res1 = _test_doc(ds1, vocabulary, params_model, params_sim, known_authors)
        res1['doc_tested'] = doc
        res1['len_doc_tested'] = len(ds1[ds1.author == 'TEST'])
        res = res.append(res1, ignore_index=True)    
    return res
        

def _gen_doc_corpus_pairs(ds, params) :
    """
    document-corpus pair generator
    
    """
    
    n = params['n']
    random = params.get('random', False)
    sampling_method = params.get('sampling_method', False)
    contig = params.get('sample_contiguous', False)
    replace = params.get('sample_w_replacements', True)
    k_docs = params.get('k_docs')
    
    ds_doc = ds[ds.author == 'doc0']
    
    if sampling_method == 'feature' :
        ln = len(ds_doc) # not working
        smp_pool = ds.index

    if sampling_method == 'verse' : # not working
        ln = len(ds_doc.verse.unique())
        smp_pool = ds.verse.unique()

    if sampling_method == 'doc_id' :
        ln = len(ds_doc.doc_id.unique())
        smp_pool = ds.doc_id.unique()
    
    ds1 = ds.copy().set_index(sampling_method)
    nmax = min(n, len(smp_pool))
    for i in range(nmax) :
        if random :
            if contig :
                k = np.random.randint(1, len(smp_pool)-ln) #sample contigous part
                smp = smp_pool[k:k+ln]
            else :
                smp = np.random.choice(smp_pool, size = ln, replace=replace)
        else :
            smp = smp_pool[i]
        
        ds_doc = ds1.loc[smp, :].reset_index()
        ds_corp = ds1.drop(smp, axis=0).reset_index()
        
        yield (ds_doc, ds_corp)


def _test_doc_corpus(ds_doc, ds_corp, params_sim, similarity_func) :
    """
    First compute similarity of doc and corpus;
    then run over  many doc-corpus pairs based on params_sim
    
    Args:
    -----
    df_corpus : corpus
    df_doc : tested corpus
    params_sim : dictionary of parameters
    similarity_func :
        vocabulary : pd.DataFrame
        params_model : dictionary of model parameters

    Returns:
    -------
    res : pd df summarizing result of each iteration

    NOTE: it is assumed that df_doc is not part of df_corpus
    """
    
    res = pd.DataFrame()
    ds_corp.loc[:,'author'] = 'corpus0'
    ds_doc.loc[:,'author'] = 'doc0'

    #ds1 = ds1.reset_index()

    res1 = _compare_doc_corpus(ds_doc, ds_corp, **similarity_func)
    res1['itr'] = 0
    res = res.append(res1, ignore_index=True)

    ds = ds_corp.append(ds_doc)
    # sample from combined corpus
        
    itr = 0
    for d in _gen_doc_corpus_pairs(ds, params_sim) :
        itr += 1
        ds_doc, ds_corp = d[0], d[1]
        ds_corp['author'] = 'corpus_smp'
        ds_doc['author'] = 'doc_smp'
        
        res1 = _compare_doc_corpus(ds_doc, ds_corp, **similarity_func)
        res1['itr'] = itr
        res1['smp_len'] = len(ds_doc)
        res = res.append(res1, ignore_index=True)
        
    return res
    
    
def _compare_doc_corpus(ds_doc, ds_corp, vocabulary, params_model) :
    md = build_model(ds_corp, vocabulary, params_model)
    res = model_predict(ds_doc, md)
    return res
