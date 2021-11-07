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

from dask.distributed import Client, progress


def bs_main(data, params_bs, vocabulary, params_model, params_sim,
         known_authors) :
    """
    Run full experiment after sampling original dataset 
    (each row is a feature) with replacements.
    
    Returns:
    res     :       original sim_full output with additional iteration 
                    indicator
    """
    
    res = pd.DataFrame()
    for itr in range(params_bs['nBS']) :
        data_bs = data.sample(n=len(data), replace=True)
        
        res1 = sim_full(data_bs, vocabulary, params_model,
        params_sim, known_authors)
        res1['itr_BS'] = itr
        res = res.append(res1, ignore_index=True)
    return res
     
#import warnings
#warnings.filterwarnings("error")

def bs_main_dist(data, params_bs, vocabulary,
                 params_model, params_sim, known_authors) :
    """
    Run full experiment after sampling original dataset (each row is a feature)
     with replacements.
    
    Returns:
    res    :    original sim_full output with additional iteration indicator
    """
        
    def func(i) :
        data_bs = data.sample(n=len(data), replace=True)
        res1 = sim_full(data_bs, vocabulary, params_model,
        params_sim, known_authors)
        res1['itr_BS'] = i
        print(i)
        return res1
    
    client = Client()
    logging.info("********************************************")
    logging.info("Dask client info: {client}")
    logging.info("************************************************")

    
    logging.info("Using Dask...")
    fut = client.map(func, range(params_bs['nBS']))
    progress(fut)
    res = client.gather(fut)
    client.close()
    return pd.concat(res)


