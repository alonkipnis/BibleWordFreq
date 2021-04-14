# pipeline: reporting
# project: bib-scripts

import pandas as pd
import numpy as np
import logging

from typing import Dict, List
from biblical_scripts.pipelines.data_science.nodes import (_prepare_data)

#import warnings
#warnings.filterwarnings("error")

def _add_stats_BS(data : pd.DataFrame, value : str, by : List) -> pd.DataFrame :
    """
    mean, std and CI's over many iterations.
    """
    grp = data.groupby(by)
    res = grp.agg({value : ['mean','std',
                               lambda x : pd.Series.quantile(x, q=.05),
                               lambda x : pd.Series.quantile(x, q=.95)
                              ]}, as_index=False).reset_index()

    res[f'{value}_mean'] = res[(value, 'mean')]
    res[f'{value}_std'] = res[(value, 'std')]
    res[f'{value}_CI05'] = res[(value, '<lambda_0>')]
    res[f'{value}_CI95'] = res[(value, '<lambda_1>')]
    res = res.drop(value, axis=1, level=0)
    return res

def _prepare_res(res) :
    """
    Convert `sim_null' results to standarad similarity results by filtering out
    non genuine docs and renaming some columns.
    """
    df = res[res.author == 'doc0'].drop('author', axis=1)
    return df.rename(columns = {'corpus' : 'wrt_author', 'true_author' : 'author'})


def report_sim(sim_null_res, vocabulary,
        params_model, params_report) -> pd.DataFrame :
    """
    Report accuracy of min-discrepancy authorship attirbution
    """
    res = _prepare_res(sim_null_res)
    res = res[res.author.isin(params_report['known_authors'])]
    df = evaluate_accuracy(res, params_report)
    df = df[df.len >= params_report['min_length_to_report']]
    logging.info(f"Accuracy = {df.succ.mean()}")
    return df
    



def evaluate_accuracy(df : pd.DataFrame, params_report) -> pd.DataFrame :
    """
    Indicate whetehr minimal discripancy is obtained by the true author.
    
    Args:
    df      data of discripancy results in columns 'value'. Othet columns
            are 'doc_id', 'author', 'wrt_author'
    params_report       parameters
    
    Returns:
    res     one row per doc_id. Indicate whether minimal discripancy is           obtained by the true author.
    """
    
    def _eval_succ(df) :
        idx_min = df.groupby(['doc_id', 'author'])['value'].idxmin()
        res_min = df.loc[idx_min, :].rename(columns={'wrt_author' : 'most_sim'})
        res_min.loc[:, 'succ'] = res_min.author == res_min.most_sim
        return res_min

    value = params_report['value']
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1 = df1.reset_index()

    res = _eval_succ(df1)
    res['param'] = str(params_report)
    return res

def report_sim_BS(sim_null_res, vocabulary,
                params_model, params_report) -> pd.DataFrame :
    """
    Report accuracy of min-discrepancy authorship attirbution
    """

