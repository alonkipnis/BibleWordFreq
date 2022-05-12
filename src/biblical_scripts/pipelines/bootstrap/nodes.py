# pipeline: bootstrap
# project: bib-scripts

"""
Note: this pipeline loads Dask-Distributed pacakge 
to accelerate computing. You can remove these 
sependencies if you are not calling 'bs_main_dist' 
"""


import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import logging

from typing import Dict, List
from biblical_scripts.pipelines.sim_full.nodes import sim_full
from biblical_scripts.pipelines.sim_val.nodes import cross_validation

from dask.distributed import Client, progress, LocalCluster
from biblical_scripts.pipelines.reporting.nodes import comp_probs, summarize_probs


def bs_main_full(data, params_bs, params_vocab,
                 params_model, params_sim_full,
                 known_authors, reference_data):
    """
    Run basic experiment after sampling original dataset
    (each row is a feature) with replacements.
    
    Returns:
    res     :       original sim_full output with additional iteration 
                    indicator
    """

    res = pd.DataFrame()
    for itr in range(params_bs['nBS']) :
        data_bs = data.sample(n=len(data), replace=True)
        res1 = sim_full(data_bs, params_vocab, params_model,
        params_sim_full, known_authors, reference_data)
        res1['itr_BS'] = itr
        res = res.append(res1, ignore_index=True)
    return res

def bs_main_val(data, params_bs, params_vocab, params_model):
    """
    Run CV experiment after sampling original dataset
    (each row is a feature) with replacements.
    
    Returns:
    res     :       original sim_full output with additional iteration 
                    indicator
    """

    res = pd.DataFrame()
    for itr in tqdm(range(params_bs['nBS'])) :
        data_bs = data.sample(n=len(data), replace=True)
        res1 = cross_validation(data_bs, params_vocab, params_model)
        res1['itr_BS'] = itr
        res = res.append(res1, ignore_index=True)
    return res
     

def bs_main_full_dask(data, params_bs, params_vocab,
                 params_model, params_sim_full,
                 known_authors, report_params,
                 reference_data):
    """
    Run full experiment after sampling original dataset (each row is a feature)
     with replacements.
    
    Returns:
    res    :    original sim_full output with additional iteration indicator
    """
    
    pd.options.mode.chained_assignment = None

    def func(i) :
        data_bs = data.sample(n=len(data), replace=True)
        res_itr = sim_full(data_bs, params_vocab, params_model,
        params_sim_full, known_authors, reference_data)

        probs = comp_probs(res_itr, report_params)
        res = summarize_probs(probs, report_params)
        res['itr_BS'] = i
        return pd.DataFrame(res, index=[0])

    cluster = LocalCluster(
        n_workers=3,
        memory_limit="4GB",
        threads_per_worker=1,
        processes=True,
    )
    client = Client(cluster)
    logging.info("********************************************")
    logging.info(f"Dask client info: {client}")
    logging.info("************************************************")

    
    logging.info("Using Dask...")
    fut = client.map(func, range(params_bs['nBS']))
    progress(fut)
    res = client.gather(fut)
    client.close()
    return pd.concat(res)


def bs_main_full(data, params_bs, params_vocab,
                 params_model, params_sim_full,
                 known_authors, report_params,
                 reference_data):
    """
    Run full experiment after sampling original dataset (each row is a feature)
     with replacements.

    Returns:
    res    :    original sim_full output with additional iteration indicator
    """

    pd.options.mode.chained_assignment = None

    def func(i):
        data_bs = data.sample(n=len(data), replace=True)
        res_itr = sim_full(data_bs, params_vocab, params_model,
                           params_sim_full, known_authors, reference_data)

        probs = comp_probs(res_itr, report_params)
        res = summarize_probs(probs, report_params)
        res['itr_BS'] = i
        return pd.DataFrame(res, index=[0])

    res = pd.DataFrame()
    for i in range(params_bs['nBS']):
        r = func(i)
        res = res.append(r, ignore_index=True)
    print(res)
    return res
