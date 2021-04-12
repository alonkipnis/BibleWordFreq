# pipeline: bootstrapping
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import logging

from typing import Dict, List
from biblical_scripts.pipelines.data_science1.nodes import (build_reduced_vocab, build_model, model_predict, _prepare_data)

def _compare_doc_corpus(ds_doc, ds_corp, vocabulary, params_model) :
    md = build_model(ds_corp, vocabulary, params_model)
    res = model_predict(ds_doc, md)
    return res

def sim_null(data, vocabulary, params_model, params_sim, known_authors) :
    """
    For each document and each corpus, check the empirical distribution of
    HC-dist(doc, corpus). Do so by joining the document to the corpus and sample from the joint corpus.
    """

    ds = _prepare_data(data)
    
    res = pd.DataFrame()
    
    lo_corps = ds.author.unique()
    lo_docs = ds.doc_id.unique()
    
    similarity_func = {'vocabulary' : vocabulary,
                'params_model' : params_model}
    
    for corp in tqdm(lo_corps) :
        logging.info("Evaluating against {auth1}")
        for doc in lo_docs :
            logging.info("Evaluating {doc}")
            ds_corp = ds[(ds.author == corp) & (ds.doc_id != doc)]
            ds_doc = ds[ds.doc_id == doc]
            res1 = _sim_doc_corpus(ds_doc, ds_corp, params_sim, similarity_func)
            res1['corpus'] = corp
            res1['doc'] = doc
            
            res = res.append(res1, ignore_index=True)
            
    return res


def _gen_sim_doc_corpus_pairs(ds, params_sim) :
    """
    document-corpus pair generator
    
    """
    
    n = params_sim['n']
    sampling_method = params_sim['sampling_method']
    contig = params_sim['sample_contiguous']
    replace = params_sim['sample_w_replacements']
    k_docs = params_sim['k_docs']
    random = params_sim['random']
    
    ds_doc = ds[ds.author == 'doc0']
    
    if sampling_method == 'feature' :
        ln = len(ds_doc)
        smp_pool = ds.index.tolist()

    if sampling_method == 'verse' :
        ln = len(ds_doc.verse.unique())
        smp_pool = ds.verse.unique()

    if sampling_method == 'doc_id' :
        ln = len(ds_doc.doc_id.unique())
        smp_pool = ds.doc_id.unique()
    
    
    for i in range(n) :
        if random :
            if contig :
                k = np.random.randint(1, len(smp_pool)-ln) #sample contigous part
                smp = smp_pool[k:k+ln]
            else :
                smp = np.random.choice(smp_pool, size = ln, replace=replace)

        ds1 = ds.copy().set_index(sampling_method)
        
        ds_doc = ds1.loc[smp.tolist(), :].reset_index()
        ds_corp = ds1.drop(smp.tolist(), axis=0).reset_index()
        
        yield (ds_doc, ds_corp)


def _sim_doc_corpus(ds_doc, ds_corp, params_sim, similarity_func) :
    """
    Compute similarity of doc and corpus; simulate many
    doc-corpus pairs based on params_sim
    
    
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
    for d in _gen_sim_doc_corpus_pairs(ds, params_sim) :
        itr += 1
        ds_doc, ds_corp = d[0], d[1]
        ds_corp['author'] = 'corpus_smp'
        ds_doc['author'] = 'doc_smp'
        
        res1 = _compare_doc_corpus(ds_doc, ds_corp, **similarity_func)
        res1['itr'] = itr
        res1['doc_len'] = len(ds_doc)
        res = res.append(res1, ignore_index=True)
        
    return res
    
    
def _evaluate_acc(res, nBs) :
    """
    Probably evaluates accuracy of method on known results
    
    Note: we should not use the method to attribute authorship, only
    asses the potential of authorship
    """
    succ = []
    for i in tqdm(range(nBS)) :
        for auth in lo_known_authors :
            res1 = test_chunck(ds1, author = auth, chunk_size = 50, sampling_method='verse')
            res1.loc[:,'test_id'] = auth + '-'+ str(i)
            res1.loc[:,'true_author'] = auth
            res = res.append(res1, ignore_index=True)
            succ += [res1.set_index('wrt_author')[value].idxmin() == auth]

    tt = res.groupby('test_id')
    idx = tt[value].idxmin()
    acc = np.mean(res.loc[idx].wrt_author == res.loc[idx].true_author)
    logging.info("Accuracy = ", acc)
    return acc


def test_chunck(ds, author, chunk_size, sampling_method='verse') :
    ds1 = ds.copy()
    lo_authors = ds.author.unique()
    ds1 = ds1.set_index(sampling_method)
    pool = ds1[ds1.author == author].index.unique()

    k = np.random.randint(1, len(pool)-chunk_size) #sample contigous part
    # TODO: arrange in the right order (not alphabetically)
    smp = pool[k:k+chunk_size]
    ds1.loc[smp,'author'] = '<TEST>'
    ds1.loc[smp,'doc_id'] = '<TEST>'
    ds1.author.unique()

    md = AuthorshipAttributionDTM(ds1.reset_index(), min_cnt = MIN_CNT,
                vocab=most_freq, gamma = GAMMA, verbose = False)
    #res = md.comp

    res = md.compute_inter_similarity(authors=['<TEST>'], wrt_authors=lo_authors)
    return res.dropna(axis=1)

