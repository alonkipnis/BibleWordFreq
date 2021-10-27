# pipeline: reporting
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
import scipy

from typing import Dict, List
from biblical_scripts.pipelines.sim.nodes import (_prepare_data)

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
    res['nBS'] = (data['itr_BS'].max()+1)
    return res

def add_stats_BS(data : pd.DataFrame, params) :
    value = params['value']
    return _add_stats_BS(_arrange_metadata(data, value), value='value', by=['doc_tested', 'corpus'])


def report_sim_full(sim_full_res, params_report) -> pd.DataFrame :
    """
    Report accuracy of min-discrepancy authorship attirbution of full evaluations
    """
    res = _arrange_metadata(sim_full_res, params_report['value']) # add 'author' and 'corpus' columns
    res = res[res.kind == 'org'] # only measuerements of original docs 
    
    res = res[res.author.isin(params_report['known_authors'])]
    res = res[res.corpus.isin(params_report['known_authors'])]
    
    df = evaluate_accuracy(res, params_report)
    df = df[df.len >= params_report['min_length_to_report']]
    logging.info(f"Accuracy = {df.succ.mean()}")
    return df
    
def evaluate_accuracy(df : pd.DataFrame, params_report) -> pd.DataFrame :
    """
    Indicate whetehr minimal discripancy is obtained by the true author.
    
    Args:
    df      data of discripancy results in columns 'value'. Othet columns
            are 'doc_id', 'author', 'corpus'
    params_report       parameters
    
    Returns:
    res     one row per doc_id. Indicate whether minimal discripancy is           obtained by the true author.
    """
    
    def _eval_succ(df) :
        idx_min = df.groupby(['doc_id', 'author'])['value'].idxmin()
        res_min = df.loc[idx_min, :].rename(columns={'corpus' : 'most_sim'})
        res_min.loc[:, 'succ'] = res_min.author == res_min.most_sim
        return res_min

    res = _eval_succ(df.reset_index())
    return res

def _comp_probs(df : pd.DataFrame, by : List) -> pd.DataFrame :
    """
    Computes mean, std, CI's, rank and t-testing for each document over 
    each corpus (as set by 'by')
    
    Args:
    df      contains similarity results
    by      list of columns to index by
    """
    
    df.loc[:,'rank'] = df.groupby(by)['value'].transform(pd.Series.rank, method='min')

    df0 = df[df.kind == 'org']
    df1 = df[df.kind != 'org']

    #df1['corpus'] = df1['corpus'].str.extract(r'([A-Za-z0-9 ]+)-ext')[0]
    grp = df1.groupby(by)
    value = 'value'
    res = grp.agg({value : ['mean', 'std', 'count', 
                               lambda x : pd.Series.quantile(x, q=.05),
                               lambda x : pd.Series.quantile(x, q=.95)
                              ]}, as_index=False).reset_index()\
        .rename(columns = {'<lambda_0>' : 'CI05', '<lambda_1>' : 'CI95'})
    dfm = df0.merge(res[['doc_tested', 'corpus', 'value']],
                      on=['doc_tested', 'corpus'], how='right')

    mu = dfm[(value, 'mean')]
    std = dfm[(value ,'std')]
    n = dfm[(value, 'count')]

    dfm.loc[:,'prob'] = 1 - (np.floor(dfm['rank'])-1) / n
    dfm.loc[:,'t-score'] = dfm[value] - mu / (std * np.sqrt(n/(n-1)))
    dfm.loc[:,'t-test'] = scipy.stats.t.sf(dfm['t-score'], df=n-1)
    return dfm

def comp_probs(sim_full_res, params_report) :
    df = _arrange_metadata(sim_full_res, params_report['value'])
    if len(df) == 0 :
        logging.error("No rows were loaded. Perhaps you did not run sim_full with the new measure?")

    dfm = _comp_probs(df, by=['author', 'doc_tested', 'corpus'])
    return dfm
    
def report_probs(dfm, params_report) :
    """
    Arrange dfm as a table 
    """
    value = params_report['value']
    dfm = dfm.rename(columns = {'value' : value})
    return dfm.pivot('corpus', 'doc_tested', [value, 'prob', 't-test', 'rank', 't-score'])

def _arrange_metadata(df, value) :
    """
    adds 'corpus' and 'author' column to evaluation results
    """
    
    df = df[df.variable.str.contains(value)]
    df.loc[:,'corpus'] = df.variable.str.extract(rf"(^[A-Za-z0-9 ]+)(-ext)?:([A-Za-z]+)")[0]
    df.loc[:,'author'] = df.doc_tested.str.extract(r"by (.+)")[0]
    df.loc[:,'variable'] = value
    return df

def report_table(df, report_params) :
    """
    df is the result of 
    """
    value = report_params['value']
    df = _arrange_metadata(df, value)
    
    known_authors = report_params['known_authors']
    df1 = df.copy()
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]    
    #df1['corpus'] = df1['variable'].str.extract(r'([^:]+):')
    
    return _report_table(df1)

def report_table_full(sim_res_full, params_report) :
    """
    Output table indicating accuracy of attribution
    
    sim_res_full has the 'itr' column
    """
    res0 = sim_res_full[sim_res_full.kind == 'org']
    return report_table(res0, params_report)

def _report_table(df) :
    """
    Output table indicating accuracy of attribution
    
    df     has columns: 'doc_id', 'author', 'corpus', 'value'
    """
    
    res_tbl = df.pivot('corpus','doc_id','value')
    lo_corpora = df.corpus.unique().tolist()
    cmin = res_tbl.idxmin().rename('min_corpus')
    res_tbl = res_tbl.append(cmin)
    res_tbl.loc['author', :] = [r[1] for r in res_tbl.columns.str.split(' by ')]
    res_tbl.loc['succ', :] = res_tbl.loc['min_corpus',:] == res_tbl.loc['author',:]
    
    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ'],:].mean(1)
    return res_tbl


def report_sim_BS(sim_full_res, vocabulary,
                params_model, params_report) -> pd.DataFrame :
    """
    Report accuracy of min-discrepancy authorship attirbution
    """

