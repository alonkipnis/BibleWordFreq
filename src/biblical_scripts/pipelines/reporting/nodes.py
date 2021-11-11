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
    res = res[res.kind == 'generic'] # only measuerements of original docs 
    
    res = res[res.author.isin(params_report['known_authors'])]
    res = res[res.corpus.isin(params_report['known_authors'])]
    
    df = evaluate_accuracy(res)
    df = df[df.len >= params_report['min_length_to_report']]
    logging.info(f"Accuracy = {df.succ.mean()}")
    return df
    

def _eval_succ(df) :
    """
    Indicate whetehr minimal discripancy is obtained by the true author.
    """
    idx_min = df.groupby(['doc_id', 'author'])['value'].idxmin()
    res_min = df.loc[idx_min, :].rename(columns={'corpus' : 'most_sim'})
    res_min.loc[:, 'succ'] = res_min.author == res_min.most_sim
    return res_min


def evaluate_accuracy(df : pd.DataFrame) -> pd.DataFrame :
    """
    Indicate whetehr minimal discripancy is obtained by the true author.
    
    Args:
    df      data of discripancy results in columns 'value'. Othet columns
            are 'doc_id', 'author', 'corpus'
    
    Returns:
    res     one row per doc_id. Indicate whether minimal discripancy is  
             obtained by the true author.
    """
    
    res = _eval_succ(df.reset_index())
    return res

def _comp_probs(df : pd.DataFrame, by : List) -> pd.DataFrame :
    """
    Computes mean, std, CI's, rank and t-test for each document over 
    each corpus (as set by 'by' parameter)
    
    Args:
    df      similarity results
    by      list of columns to index by
    """
    
    df.loc[:,'rank'] = df.groupby(by)['value'].transform(pd.Series.rank, method='min')

    df0 = df[df.kind == 'generic']
    df1 = df[df.kind != 'generic']

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
        logging.error("No results were found. Perhaps you did not run"
                      " sim_full with the requested measure?")

    dfm = _comp_probs(df, by=['author', 'doc_tested', 'corpus'])
    return dfm
    
def report_probs(dfm, params_report) :
    """
    Arrange dfm as an easy-to-read table 

    """
    value = params_report['value']
    dfm = dfm.rename(columns = {'value' : value})
    return dfm.pivot('corpus', 'doc_tested', 
        [value, 'prob', 't-test', 'rank', 't-score'])

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
    
    return _report_table(df1)


def _report_table(sim_res) :
    """
    Arrange discrepancies test results to indicate accuracy of 
    authorship attribution
    """
    res_tbl = sim_res.pivot('corpus','doc_id','value')
    lo_corpora = sim_res.corpus.unique().tolist()
    cmin = res_tbl.idxmin().rename('min_corpus')
    res_tbl = res_tbl.append(cmin)
    res_tbl.loc['author', :] = [r[1] for r in res_tbl.columns.str.split(' by ')]
    res_tbl.loc['succ', :] = res_tbl.loc['min_corpus',:] == res_tbl.loc['author',:]
    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ'],:].mean(1)
    return res_tbl


def report_table_known(df, report_params) :
    """
    Arrange discrepancies test results to indicate accuracy of 
    authorship attribution of known authors
    """
    value = report_params['value']
    known_authors = report_params['known_authors']
    
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:,'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1 = df1[df1.corpus.isin(known_authors)]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]
    df1 = df1[df1['author'].isin(known_authors)]  # bc its 'known_authors_only'
    
    df_res = _report_table(df1)
    print("\n \t MEAN: ", df_res['mean'],"\n\n")
    return df_res

def report_table_unknown(df, report_params) :
    
    value = report_params['value']
    known_authors = report_params['known_authors']
    
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:,'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1 = df1[df1.corpus.isin(known_authors)]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]
    df1 = df1[~df1['author'].isin(known_authors)] # bc its 'non known_authors_only'
    
    return _report_table(df1)

def report_table_len(df, params_report) :
    """
    Output table indicating accuracy of attribution
    for results obtained from pipeline chunk_len
    
    Here we need to group by chunk_length and 
    average over iterations and authors 

    """

    value = params_report['value']
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:,'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1['author'] = df1['true_author']
    df1['doc_id'] = df1['experiment'] + ":" + df1['true_author'] \
                + ":" + df1['itr'].astype(str) + ":" + df1['chunk_size'].astype(str)
    df1 = df1.reset_index()
    
    df_res = _eval_succ(df1)

    # average over chunk_len
    df_res['succ'] = df_res['succ'] + .0
    grp = df_res.groupby('chunk_size')
    res = grp.agg({'succ' : ['mean']}, as_index=False).reset_index()

    res[f'succ_mean'] = res[('succ', 'mean')]
    res = res.drop('succ', axis=1, level=0)

    return res

def _pre_report_table_full(df) :
    """
    Compute rank-based P-values w.r.t. each 
    corpus

    """

    lo_docs = df.doc_tested.unique().tolist()
    res = pd.DataFrame()
    for doc in lo_docs :
        df1 = df[df.doc_tested == doc]
        df1.loc[:, 'rnk'] = df1.groupby('corpus')['value'].rank() 
        num_of_docs = df1.groupby('corpus')['value'].transform('count')
        df1.loc[:, 'rnk_pval'] = 1 - df1.loc[:, 'rnk'] / num_of_docs
        df2 = df1[df1.kind == 'generic']
        res = res.append(df2, ignore_index=True)
    return res


def report_table_full_known(sim_res_full, params_report, known_authors) :
    
    value = params_report['value']
    res = _arrange_metadata(sim_res_full, value)
    res = _pre_report_table_full(res)


    sig_level = params_report['sig_level']

    res_f=res[res.author.isin(known_authors)]
    lo_authors = res_f.author.unique().tolist()
    lo_corpora = res_f.corpus.unique().tolist()

    res_tbl = res_f.pivot('corpus', 'doc_id', 'rnk_pval')
    
    cmin = res_tbl.idxmax().rename('max_pval_corpus')
    res_tbl = res_tbl.append(cmin)

    res_tbl.loc['author', :] = [r[1] for r in res_tbl.columns.str.split(' by ')]
    res_tbl.loc['succ', :] = res_tbl.loc['max_pval_corpus',:] == res_tbl.loc['author',:]


    # add length info
    #res_tbl = res_tbl.T.merge(res_f[['doc_id', 'len']].drop_duplicates(), on='doc_id').T

    #compute false alarm rate
    res_tbl.loc['false_alarm', :] = False
    for auth in lo_authors :
        idcs = res_tbl.loc['author',:] == auth
        res_tbl.loc['false_alarm', idcs] = res_tbl.loc[auth, idcs] < sig_level

    # add column indicateing success and false alarm rates
    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ', 'false_alarm'],:].mean(1)

    return res_tbl

def report_table_full_unknown(sim_res_full, params_report, unknown_authors) :
    
    value = params_report['value']
    res = _arrange_metadata(sim_res_full, value)
    res = _pre_report_table_full(res)

    sig_level = params_report['sig_level']

    res_f=res[res.author.isin(unknown_authors)]
    lo_corpora = res.corpus.unique().tolist()

    res_tbl = res_f.pivot('corpus', 'doc_id', 'rnk_pval')
    cmin = res_tbl.idxmax().rename('max_pval_corpus')
    res_tbl = res_tbl.append(cmin)

    res_tbl.loc['author', :] = [r[1] for r in res_tbl.columns.str.split(' by ')]
    res_tbl.loc['significane', :] = res_tbl.loc[lo_corpora,:].max() > sig_level

    return res_tbl

def _report_table(df) :
    """
    Output table indicating accuracy of attribution based
    on discrepancies values
    
    Args:
    -----
    df     has columns: 'doc_id', 'author', 'corpus', 'value'

    Returns:
    -------
    res_tbl     indicates whether the attribution of each document
                is correct, as well as the overal accuracy which 
                is the average of the indicator function of correctness
    """
    
    res_tbl = df.pivot('corpus','doc_id','value')
    lo_corpora = df.corpus.unique().tolist()
    cmin = res_tbl.idxmin().rename('min_corpus')
    res_tbl = res_tbl.append(cmin)
    res_tbl.loc['author', :] = [r[1] for r in res_tbl.columns.str.split(' by ')]
    res_tbl.loc['succ', :] = res_tbl.loc['min_corpus',:] == res_tbl.loc['author',:]
    
    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ'],:].mean(1)
    return res_tbl


