# pipeline: reporting
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
import scipy

from typing import Dict, List
from biblical_scripts.pipelines.sim.nodes import (_prepare_data)
from biblical_scripts.pipelines.data_engineering.nodes import add_convert


# import warnings
# warnings.filterwarnings("error")

def _add_stats_BS(data: pd.DataFrame, value: str, by: List) -> pd.DataFrame:
    """
    mean, std and CI's over many iterations.
    """
    grp = data.groupby(by)
    res = grp.agg({value: ['mean', 'std',
                           lambda x: pd.Series.quantile(x, q=.05),
                           lambda x: pd.Series.quantile(x, q=.95)
                           ]}, as_index=False).reset_index()

    res[f'{value}_mean'] = res[(value, 'mean')]
    res[f'{value}_std'] = res[(value, 'std')]
    res[f'{value}_CI05'] = res[(value, '<lambda_0>')]
    res[f'{value}_CI95'] = res[(value, '<lambda_1>')]
    res = res.drop(value, axis=1, level=0)
    res['nBS'] = (data['itr_BS'].max() + 1)
    return res


def add_stats_BS(data: pd.DataFrame, params):
    value = params['value']
    return _add_stats_BS(_arrange_metadata(data, value), value='value', by=['doc_tested', 'corpus'])


def report_sim_full(sim_full_res, params_report) -> pd.DataFrame:
    """
    Report accuracy of min-discrepancy authorship attirbution of full evaluations
    """
    res = _arrange_metadata(sim_full_res, params_report['value'])  # add 'author' and 'corpus' columns
    res = res[res.kind == 'generic']  # only measuerements of original docs

    res = res[res.author.isin(params_report['known_authors'])]
    res = res[res.corpus.isin(params_report['known_authors'])]

    df = evaluate_accuracy(res)

    # Patch to report accuracy only on docs exceeding a certain length
    df = df[df.len >= params_report['min_length_to_report']]
    logging.info(f"Accuracy = {df.succ.mean()}")
    return df


def _eval_succ(df):
    """
    Indicate whetehr minimal discripancy is obtained by the true author.
    """
    idx_min = df.groupby(['doc_id', 'author'])['value'].idxmin()
    res_min = df.loc[idx_min, :].rename(columns={'corpus': 'most_sim'})
    res_min.loc[:, 'succ'] = res_min.author == res_min.most_sim
    return res_min


def evaluate_accuracy(df: pd.DataFrame) -> pd.DataFrame:
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


def _comp_probs(df: pd.DataFrame, by: List) -> pd.DataFrame:
    """
    Computes mean, std, CI's, rank and t-test for each document over 
    each corpus (as set by 'by' parameter)
    
    Args:
    df      similarity results
    by      list of columns to index by
    """

    df.loc[:, 'rank'] = df.groupby(by)['value'].transform(
        pd.Series.rank, pct=True, method='min')

    df0 = df[df.kind == 'generic']
    df1 = df[df.kind != 'generic']

    grp = df1.groupby(by)
    value = 'value'
    res = grp.agg({value: ['mean', 'std', 'count',
                           lambda x: pd.Series.quantile(x, q=.05),
                           lambda x: pd.Series.quantile(x, q=.95)
                           ]}, as_index=False).reset_index() \
        .rename(columns={'<lambda_0>': 'CI05', '<lambda_1>': 'CI95'})

    res.loc[:, 'doc_id'] = res['doc_tested']

    dfm = df0.merge(res[['doc_id', 'corpus', 'value']],
                    on=['doc_id', 'corpus'], how='left') # only include results for generic chapters

    mu = dfm[(value, 'mean')]
    std = dfm[(value, 'std')]
    n = dfm[(value, 'count')]

    # dfm.loc[:,'prob'] = 1 - (np.floor(dfm['rank'])-1) / n
    dfm.loc[:, 'rank_pval'] = 1 - dfm['rank']
    dfm.loc[:, 't-score'] = dfm[value] - mu / (std * np.sqrt(n / (n - 1)))
    dfm.loc[:, 't_pval'] = scipy.stats.t.sf(dfm['t-score'], df=n - 1)
    dfm.loc[:, 'author'] = dfm.author.apply(_get_author_from_doc_id)
    return dfm


def comp_probs(sim_full_res, params_report):
    """
    Rank-based test and t-test for discrepancy results obtained by 
    augmanting each corpus with the tested document
    """

    sim_full_res = _arrange_sim_full_results(sim_full_res)
    df = _arrange_metadata(sim_full_res, params_report['value'])

    if len(df) == 0:
        logging.error("No results were found. Perhaps you did not run"
                      " sim_full with the requested measure?")

    dfm = _comp_probs(df, by=['author', 'doc_tested', 'corpus'])
    return dfm


def report_probs(dfm, params_report):
    """
    Arrange dfm as an easy-to-read table 

    """
    value = params_report['value']
    dfm = dfm.rename(columns={'value': value})
    return dfm.pivot('corpus', 'doc_tested',
                     [value, 'rank_pval', 't_pval', 't-score']).reset_index()


def summarize_probs(dfm, params, chapters_to_report):
    """
    Print summary from probabilities evaluated in comp_probs
    This function is mostly here to provide information for
    debugging purposes.

    """

    if not chapters_to_report.empty:
        dfm = _filter_to_certain_chapters(dfm, chapters_to_report)

    dfm.loc[:, 'sig_t'] = dfm['t_pval'] < params['sig_level']
    dfm.loc[:, 'sig_rank'] = dfm['rank_pval'] < params['sig_level']

    print("========================RESULTS================================")
    dfr = dfm[dfm.author.isin(params['known_authors'])]
    idx_var = dfr.groupby('doc_id')['value'].idxmin()
    idx_tpval = dfr.groupby('doc_id')['t_pval'].idxmax()
    print(f"Accuracy with {dfr['variable'].unique()}: ",
          (dfr.loc[idx_var, 'corpus'] == dfr.loc[idx_var, 'author']).mean())
    print(f"Accuracy with t p-value: ",
          (dfr.loc[idx_tpval, 'corpus'] == dfr.loc[idx_tpval, 'author']).mean())

    print("False Negatives with t p-value:")
    print(dfr[(dfr.author == dfr.corpus) & dfr.sig_t])
    print("False 'Positives' with t p-value:")
    print(dfr[(dfr.author != dfr.corpus) & ~dfr.sig_t])
    df_res = dfr[dfr.author == dfr.corpus]\
                .groupby(['variable']) \
                .mean() \
                .filter(['sig_t', 'sig_rank']) \
                .rename(columns={'sig_t': 'FNR-t', 'sig_rank': 'FNR-rank'})
    print(f"False negative rates over {len(dfr)} chapters of known authorship:")
    print(df_res)
    return df_res


def _arrange_metadata(df, value):
    """
    adds 'corpus' and 'author' column to evaluation results
    """

    df = df[df.variable.str.contains(value)]
    df.loc[:, 'corpus'] = df.variable.str.extract(rf"(^[A-Za-z0-9 ]+)(-ext)?:([A-Za-z]+)")[0]
    df.loc[:, 'author'] = df.doc_tested.str.extract(r"([^|]*)")[0]
    df.loc[:, 'variable'] = value
    return df


def report_table(df, report_params):
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


def _report_table(sim_res):
    """
    Arrange discrepancies test results to indicate accuracy of 
    authorship attribution
    """
    res_tbl = sim_res.pivot('corpus', 'doc_id', 'value')
    lo_corpora = sim_res.corpus.unique().tolist()
    cmin = res_tbl.idxmin().rename('min_corpus')
    res_tbl = res_tbl.append(cmin)
    res_tbl.loc['author', :] = [_get_author_from_doc_id(r) for r in res_tbl.columns]
    res_tbl.loc['succ', :] = res_tbl.loc['min_corpus', :] == res_tbl.loc['author', :]
    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ'], :].mean(1)
    print(res_table.loc['succ', :])
    return res_tbl.reset_index()


def _filter_to_certain_chapters(df, book_chapter: pd.DataFrame):
    """
    Removes from df rows not included in book_chapter
    """

    book_chapter.loc[:, 'feature'] = 'null'
    book_chapter.loc[:, 'chapter'] = book_chapter['book'] \
                                     + '.' + book_chapter['chapter'].astype(str)
    lo_chapters = _prepare_data(
        book_chapter[['chapter', 'author', 'feature']]
    ).doc_id.tolist()
    logging.info(f"Considering {len(lo_chapters)} chapters in the report.")
    return df[df.doc_id.isin(lo_chapters)]


def report_table_known(df, report_params, chapters_to_report=pd.DataFrame()):
    """
    Arrange discrepancies test results to indicate accuracy of 
    authorship attribution of known authors

    Args:
    -----
    df                  discrepancies of many (doc, cropus) pairs
    report_params       parameters indicating how to report 
    chapters_to_report   only report on chapters in this list
    """

    if not chapters_to_report.empty:
        df = _filter_to_certain_chapters(df, chapters_to_report)

    value = report_params['value']
    known_authors = report_params['known_authors']

    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:, 'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1 = df1[df1.corpus.isin(known_authors)]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]
    df1 = df1[df1['author'].isin(known_authors)]  # bc its 'known_authors_only'

    df_res = _report_table(df1)

    print("========================RESULTS================================")
    print(f"Reporting over {df_res.shape[1] - 1} chapters:")
    print("\n \t MEAN Values: ", df_res['mean'], "\n\n")
    return df_res.reset_index()


def report_table_unknown(df, report_params):
    value = report_params['value']
    known_authors = report_params['known_authors']
    unknown_authors = report_params['unknown_authors']

    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:, 'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1 = df1[df1.corpus.isin(known_authors)]
    df1 = df1.reset_index()
    df1 = df1[df1.len >= report_params['min_length_to_report']]
    df1 = df1[df1['author'].isin(unknown_authors)]

    return _report_table(df1)


def report_table_len(df, params_report):
    """
    Output table indicating accuracy of attribution
    for results obtained from pipeline chunk_len
    
    Here we need to group by chunk_length and 
    average over iterations and authors 

    """

    value = params_report['value']
    df1 = df[df['variable'].str.contains(f":{value}")]
    df1.loc[:, 'corpus'] = df1['variable'].str.extract(r'([^:]+):')[0]
    df1['author'] = df1['true_author']
    df1['doc_id'] = df1['experiment'] + ":" + df1['true_author'] \
                    + ":" + df1['itr'].astype(str) + ":" + df1['chunk_size'].astype(str)
    df1 = df1.reset_index()

    df_res = _eval_succ(df1)

    # average over chunk_len
    df_res['succ'] = df_res['succ'] + .0
    grp = df_res.groupby('chunk_size')
    res = grp.agg({'succ': ['mean']}, as_index=False).reset_index()

    res[f'succ_mean'] = res[('succ', 'mean')]
    res = res.drop('succ', axis=1, level=0)

    return res


def _pre_report_table_full(df):
    """
    Compute rank-based P-values w.r.t. each 
    corpus

    """

    lo_docs = df.doc_tested.unique().tolist()
    res = pd.DataFrame()
    for doc in lo_docs:
        df1 = df[df.doc_tested == doc]
        df1.loc[:, 'rnk'] = df1.groupby('corpus')['value'].rank(pct=True,
                                                                method='min')
        df1.loc[:, 'rnk_pval'] = 1 - df1.loc[:, 'rnk']
        df2 = df1[df1.kind == 'generic']
        res = res.append(df2, ignore_index=True)
    return res


def _arrange_sim_full_results(res):
    """
    For HC, use max(HC, 1)
    """
    min_value_hc = 1.0
    idcs = res.variable.str.contains(':HC')
    res.loc[idcs, 'value'] = np.maximum(res.loc[idcs, 'value'], min_value_hc)
    return res


def report_table_full_known(probs, params):
    """
    Params:
        :probs:     the output of compute_probs

    """

    dfr = probs[probs.author.isin(params['known_authors'])]
    value = params['value']

    assert len(dfr.variable.unique())==1, "Cannot report on more than one variable"

    dfr = dfr.rename(columns={'value' : value})
    dfm = dfr.pivot('corpus', 'doc_id', [value, 't-score'])

    idx_min = dfm.idxmin()
    succ_val = idx_min[value].index.str.extract(r"([^|]+)")[0] == idx_min[value].values.tolist()
    succ_t = idx_min['t-score'].index.str.extract(r"([^|]+)")[0] == idx_min['t-score'].values.tolist()

    dfm.loc['succ', :] = pd.concat([succ_val, succ_t], ignore_index=True).values

    return dfm



def OLD_report_table_full_known(sim_res_full, report_params, chapters_to_report):

    if not chapters_to_report.empty:
        sim_res_full = _filter_to_certain_chapters(sim_res_full, chapters_to_report)

    known_authors = report_params['known_authors']
    value = report_params['value']
    # >>HERE!! add t-test scores and report on its value
    # Alos, use idxmin instea of idxmax in Line 382
    import pdb; pdb.set_trace()
    res = _arrange_metadata(sim_res_full, value)
    res = _pre_report_table_full(res)

    res = res[res.len >= report_params['min_length_to_report']]

    res_f = res[res.author.isin(known_authors)]
    lo_authors = res_f.author.unique().tolist()
    lo_corpora = res_f.corpus.unique().tolist()

    res_tbl = res_f.pivot('corpus', 'doc_id', 'rnk_pval')

    cmin = res_tbl.idxmax().rename('max_pval_corpus')
    res_tbl = res_tbl.append(cmin)

    res_tbl.loc['author', :] = [_get_author_from_doc_id(r) for r in res_tbl.columns]
    res_tbl.loc['succ', :] = res_tbl.loc['max_pval_corpus', :] == res_tbl.loc['author', :]

    # add length info
    # res_tbl = res_tbl.T.merge(res_f[['doc_id', 'len']].drop_duplicates(), on='doc_id').T

    # compute false alarm rate
    res_tbl.loc['false_alarm', :] = False
    for auth in lo_authors:
        idcs = res_tbl.loc['author', :] == auth
        res_tbl.loc['false_alarm', idcs] = res_tbl.loc[auth, idcs] < report_params['sig_level']

    # add column indicating success and false alarm rates
    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ', 'false_alarm'], :].mean(1)

    print(f"Reporting over {res_tbl.shape[1] - 1} chapters:")
    print("\n \t MEAN: ", res_tbl['mean'], "\n\n")
    return res_tbl.reset_index()


def report_table_full_unknown(sim_res_full, params_report):
    dfr = probs[~probs.author.isin(params['known_authors'])]
    value = params['value']

    assert len(dfr.variable.unique()) == 1, "Cannot report on more than one variable"

    dfr = dfr.rename(columns={'value': value})
    dfm = dfr.pivot('corpus', 'doc_id', [value, 't-score'])

    idx_min = dfm.idxmin()
    succ_val = idx_min[value].index.str.extract(r"([^|]+)")[0] == idx_min[value].values.tolist()
    succ_t = idx_min['t-score'].index.str.extract(r"([^|]+)")[0] == idx_min['t-score'].values.tolist()

    dfm.loc['succ', :] = pd.concat([succ_val, succ_t], ignore_index=True).values

    return dfm


def _get_author_from_doc_id(st: str) -> str:
    return st.split('|')[0]

def _report_table(df):
    """
    Output table indicating accuracy of attribution based
    on discrepancies values
    
    Params:
    :df:     DataFrame with columns: 'doc_id', 'author', 'corpus', 'value'

    Returns:
    :res_tbl:    indicates whether the attribution of each document
                is correct, as well as the overall accuracy which
                is the average of the indicator function of correctness
    """

    res_tbl = df.pivot('corpus', 'doc_id', 'value')
    lo_corpora = df.corpus.unique().tolist()
    cmin = res_tbl.idxmin().rename('min_corpus')
    res_tbl = res_tbl.append(cmin)
    res_tbl.loc['author', :] = [_get_author_from_doc_id(r) for r in res_tbl.columns]
    res_tbl.loc['succ', :] = res_tbl.loc['min_corpus', :] == res_tbl.loc['author', :]
    df_len = df.filter(['doc_id', 'len']).drop_duplicates().set_index('doc_id')
    res_tbl.loc['len', :] = df_len['len']

    res_tbl['mean'] = res_tbl.loc[lo_corpora + ['succ'] + ['len'], :].mean(1)
    return res_tbl


def report_vocab(df_vocabulary, data) -> pd.DataFrame:
    df = add_convert(df_vocabulary, data)
    return df
