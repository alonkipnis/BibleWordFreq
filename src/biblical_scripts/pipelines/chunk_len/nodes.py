"""
This is a boilerplate pipeline 'chunk_len'
generated using Kedro 0.17.0
"""

from tqdm import tqdm
from typing import List, Dict
import pandas as pd
from biblical_scripts.pipelines.sim.nodes import (
    build_model, model_predict)
import logging
import numpy as np
from biblical_scripts.pipelines.sim_full.nodes import _test_doc, _gen_test_doc, _check_doc


def _sample_chunk(ds, author, chunk_size,
                  sampling_method='verse', contiguous=True):
    """
	sample a chunk of size 'chunk_size' from data by author 'author'.
	Mark sampled chunk as '<TEST>'
	"""

    ds1 = ds.set_index(sampling_method)
    pool = ds1[ds1.author == author].index.unique()  # create a pool from

    # given author's data
    assert(2*len(pool) > chunk_size)

    def _sample_contiguous(pool, chunk_size):
        # randomly sample a contiguous part of size 'chunk_size'
        # from 'pool'
        k = np.random.randint(0, max(len(pool) - chunk_size, 1))
        return pool[k:k + chunk_size]

    def _sample_random(pool, chunk_size):
        # randomly sample a contigous part of size 'chunk_size'
        # from 'pool'
        return np.random.choice(pool, chunk_size, replace=True)

    if contiguous:
        smp = _sample_contiguous(pool, chunk_size)
    else:
        smp = _sample_random(pool, chunk_size)

    ds1.loc[smp, 'author'] = '<TEST>'
    ds1.loc[smp, 'doc_id'] = '<TEST>'

    return ds1


def _evaluate_discrepancies(data, vocab, model_params):
    """
	Build a word-frequency model. Evaluate discrepancy
	of document '<TEST>' with respect to any of the other
	authors in the data
	"""

    data_train = data[~(data.author == "<TEST>")]
    data_test = data[data.author == "<TEST>"]

    md, _ = build_model(data_train, vocab, model_params)
    df1 = model_predict(data_test, md)
    return df1


def _evaluate_discrepancies_full(data, vocab, model_params):
    """
	Build a word-frequency model. Evaluate discrepancy
	of document '<TEST>' with respect to any of the other
	authors in the data
	"""

    data_train = data[~(data.author == "<TEST>")]
    data_test = data[data.author == "<TEST>"]

    res = pd.DataFrame()

    res1 = _test_doc(data, vocab, model_params, params_sim, known_authors)
    res1['doc_tested'] = doc
    # Here we remove results of documents that we choose not to report on
    res1 = res1[res1.doc_id.isin(lo_chapters_to_report) | res1.doc_id.str.contains('chapter0')]
    # It may be better to remove those documents from the pool altogether

    res1['len_doc_tested'] = len(ds1[ds1.author == 'TEST'])
    return res1


def test_chunks_len(data, vocab, lo_authors,
                    model_params, chunk_len_params):
    """
	Discrepancies versus length. For each author and chunk_length, sample 
	a chunk from the corpus of that author of size chunk_length and
	evaluate discrepancy of that chunk to all corpora
	"""

    lo_lengths = chunk_len_params['chunk_lengths']
    nMonte = chunk_len_params['nMonte']

    logging.info("Evaluating discrepancies versus length of random chunks...")
    res = pd.DataFrame()
    for chunk_size in tqdm(lo_lengths):
        print(f"Checking chunk_size = {chunk_size}")
        for i in range(nMonte):
            for auth in lo_authors:
                ds1 = _sample_chunk(data.copy(), author=auth,
                                    chunk_size=chunk_size,
                                    sampling_method=chunk_len_params['sampling_method'],
                                    contiguous=chunk_len_params['contiguous_chunk']
                                    )

                res1 = _evaluate_discrepancies(ds1, vocab, model_params)
                res1.loc[:, 'true_author'] = auth
                res1.loc[:, 'chunk_size'] = chunk_size
                res1.loc[:, 'itr'] = i
                res1.loc[:, 'experiment'] = 'chunk_len'
                res = res.append(res1, ignore_index=True)
    return res
