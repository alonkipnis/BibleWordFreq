# pipeline: sim_full
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import logging

from typing import Dict, List
from biblical_scripts.pipelines.sim.nodes import (
    build_model, model_predict, _prepare_data)


def _check_doc(ds, vocabulary, params_model) -> pd.DataFrame:
    """
    Build a model from all data after removing entries with ds['author]=='TEST'
    and check the document containing entries ds['doc_id']=='TEST' against
    this model. Return the discrepancy result.

    Params:
    :ds:  data
    :vocabulary:  vocabulary to be used in model construction
    :params_model:  model parameters

    Returns:
         model prediction results (arranged as a dataframe)
    """

    md, _ = build_model(ds[ds.author != 'TEST'], vocabulary, params_model)
    return model_predict(ds[ds.author == 'TEST'], md)


def _test_doc(ds, vocabulary, params_model, params_sim, known_authors):
    """
    Test the document marked as 'TEST' against each corpus in
    known_authors
    """
    itr = 0
    res = pd.DataFrame()

    res1 = _check_doc(ds, vocabulary, params_model)  # evaluate wrt to corpus
    res1['itr'] = itr
    res1['smp_len'] = len(ds[ds.author == 'TEST'])
    res1['kind'] = 'generic'
    res = res.append(res1, ignore_index=True)


    for ds1 in _gen_test_doc(ds, known_authors, params_sim):
        # ds1 is a document from an 'extended' ('modified')
        # corpus
        itr += 1
        res1 = _check_doc(ds1, vocabulary, params_model)
        res1 = res1[res1.variable.str.contains('-ext')]
        res1['itr'] = itr
        res1['checked_doc_len'] = len(ds1[ds1.author == 'TEST'])
        res1['kind'] = 'extended'
        res = res.append(res1, ignore_index=True)
    return res


def _gen_test_doc(ds0, known_authors, params):
    """
    document-corpus pair generator for assessing probability
    of doc-corpus scores. We extend each corpus of
    'known_authors' by adding a tested document (assumed to
    have 'TEST' as its author). We then 'sample' documents
    from the extended corpus. 
    
    
    TODO:
    At the moment, the generator simply goes over all
    documents in the pool. We may want to allow instead
    sampling from the pool with repetitions, in which case
    we would need to change the probability model we use
    to assess confident.
    
    """

    #original_corpus = ds0[ds0.author == 'TEST'].doc_id.values[0].split('|')[0]
    for corpus in known_authors:
        ds = ds0.copy()

        # mark docs with author == corpus or author == TEST
        ds.loc[ds.author.isin([corpus, 'TEST']), 'author'] = f'{corpus}-ext'
        # create a pool consisting of docs exceeding minimum length
        ds_pool = ds[(ds.author == f'{corpus}-ext') & (ds.len >= params['min_length_to_consider'])]

        smp_pool = ds_pool.doc_id.unique()
        for smp in smp_pool:
            ds1 = ds.copy()
            ds1.loc[ds.doc_id == smp, 'author'] = 'TEST'
            yield ds1


def sim_full(data, vocabulary, params_model,
             params_sim, known_authors, reference_data):
    """
    report discrepancy between every document to every 
    corpus of known authorship
    """

    ds = _prepare_data(data)
    lo_docs_org = ds.doc_id.unique()

    to_report = reference_data[reference_data.to_report]
    lo_chapters_to_report = to_report['author'] + '|' + to_report['book'] + '.' + to_report['chapter'].astype(str)

    lo_docs = [doc for doc in lo_docs_org if
               doc in lo_chapters_to_report.values or 'chapter0' in doc]

    res = pd.DataFrame()
    logging.info(f"Going over a list of {len(lo_docs)} docs...")
    for doc in tqdm(lo_docs):
        logging.info(f"Testing {doc}...")
        ds1 = ds[ds.author.isin(known_authors) | (ds.doc_id == doc)]
        ds1.loc[ds1.doc_id == doc, 'author'] = 'TEST'  # mark tested doc
        res1 = _test_doc(ds1, vocabulary, params_model, params_sim, known_authors)
        res1['doc_tested'] = doc
        # Here we remove results of documents that we chosen not to report on
        # because they were too small hence their HC score is usually low
        res1 = res1[res1.doc_id.isin(lo_chapters_to_report) | res1.doc_id.str.contains('chapter0')]
        # It may be better to remove those documents from the pool altogether

        res1['len_doc_tested'] = len(ds1[ds1.author == 'TEST'])
        res = res.append(res1, ignore_index=True)
    return res


def _gen_doc_corpus_pairs(ds, params):
    """
    document-corpus pair generator: We first form a pool
     of text atoms, such as lemma, verse, or chapter. The
    'new' document is obtained by sampling from this pool.

    In the most simple situation, the pool is simply all chapters
    in the corpus.
    
    """

    n = params['n']
    random = params.get('random', False)
    sampling_method = params.get('sampling_method', False)
    contig = params.get('sample_contiguous', False)
    replace = params.get('sample_w_replacements', True)

    ds_doc = ds[ds.author == 'doc0']

    if sampling_method == 'feature':
        ln = len(ds_doc)  # not working
        smp_pool = ds.index

    if sampling_method == 'verse':  # not working
        ln = len(ds_doc.verse.unique())
        smp_pool = ds.verse.unique()

    if sampling_method == 'doc_id':
        ln = len(ds_doc.doc_id.unique())
        smp_pool = ds.doc_id.unique()

    ds1 = ds.copy().set_index(sampling_method)
    nmax = min(n, len(smp_pool))
    for i in range(nmax):
        if random:
            if contig:
                k = np.random.randint(1, len(smp_pool) - ln)  # sample contigous part
                smp = smp_pool[k:k + ln]
            else:
                smp = np.random.choice(smp_pool, size=ln, replace=replace)
        else:
            smp = smp_pool[i]

        ds_doc = ds1.loc[smp, :].reset_index()
        ds_corp = ds1.drop(smp, axis=0).reset_index()

        yield (ds_doc, ds_corp)


def _compare_doc_corpus(ds_doc, ds_corp, vocabulary, params_model):
    md = build_model(ds_corp, vocabulary, params_model)
    res = model_predict(ds_doc, md)
    return res
