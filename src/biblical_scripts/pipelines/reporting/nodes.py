# pipeline: reporting
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
import scipy
from scipy.stats import f as fdist

from typing import List
from biblical_scripts.pipelines.sim.nodes import (_prepare_data)
from biblical_scripts.pipelines.data_engineering.nodes import (add_convert)


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
    return _add_stats_BS(_arrange_metadata(data, value), value='value',
                         by=['doc_tested', 'corpus'])

def add_stats_BS_full(sim_full_res, params_report):
    """
    Rank-based test and t-test for discrepancy results obtained by
    augmanting each corpus with the tested document
    """

    sim_full_res = _arrange_sim_full_results(sim_full_res)
    df = _arrange_metadata(sim_full_res, params_report['value'])

    if len(df) == 0:
        logging.error("No results were found. Perhaps you did not run"
                      " sim_full with the requested measure?")

    if params_report['anova']:
        dfm = _comp_probs_anova(df)
    else:
        dfm = _comp_probs_t(df, by=['author', 'doc_tested',
                                    'corpus'],
                            log_scale=params_report['log_scale'])
    return dfm


def report_sim_full(sim_full_res, params_report) -> pd.DataFrame:
    """
    Report accuracy of min-discrepancy authorship attribution of Full evaluations
    """
    res = _arrange_metadata(sim_full_res, params_report['value'])  # add 'author' and 'corpus' columns
    res = res[res.kind == 'generic']  # only measurements of original docs

    res = res[res.author.isin(params_report['known_authors'])]

    df = evaluate_accuracy(res)

    # Patch to report accuracy only on docs exceeding a certain length
    df = df[df.len >= params_report['min_length_to_report']]
    logging.info(f"Accuracy = {df.succ.mean()}")
    return df


def _eval_succ(df):
    """
    Indicate whether minimal discrepancy is obtained by the true author.
    """
    idx_min = df.groupby(['doc_id', 'author'])['value'].idxmin()
    res_min = df.loc[idx_min, :].rename(columns={'corpus': 'most_sim'})
    res_min.loc[:, 'succ'] = res_min.author == res_min.most_sim
    return res_min


def evaluate_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicate whether minimal discrepancy is obtained by the true author.
    
    Parameters:
        :df:  data of discrepancy results in columns 'value'. Other columns
            are 'doc_id', 'author', 'corpus'
    
    Returns:
    :res:     one row per doc_id. Indicate whether minimal discrepancy is
             obtained by the true author.
    """

    res = _eval_succ(df.reset_index())
    return res


def _comp_probs_anova(df: pd.DataFrame) -> pd.DataFrame:
    """
    F-test. Numerator is sum-of-squares of inter-corpus HC
    scores when the tested document belongs to the corpus
    with which we check attribution. The denominator is the
    sum-of-squares of inter-corpus HC scores when the
    document does not belong to the corpus.
    """

    df0 = df[df.kind == 'generic']
    df1 = df[df.kind != 'generic']

    def ssquares(x):
        return np.sum((x - x.mean()) ** 2)

    df01 = df0[df0.author == df0.corpus]
    df02 = df01.filter(['corpus', 'author', 'variable', 'doc_tested', 'value'])
    dfss0 = df02.groupby(['variable', 'corpus', 'author']).agg(['mean', 'count', ssquares])

    dfss1 = df1.filter(['author', 'variable', 'value',
                        'corpus', 'doc_id', 'doc_tested']) \
        .groupby(['doc_tested', 'variable', 'corpus', 'author']) \
        .agg(['mean', 'count', ssquares])

    dfss = dfss1.join(dfss0.reset_index(2).drop('author', axis=1),
                      rsuffix='0', lsuffix='1')

    def ftest(r):
        dfn = r[('value1', 'count')] - 1
        dfd = r[('value0', 'count')] - 1
        s = r[('value1', 'ssquares')] / r[('value0', 'ssquares')]
        return fdist.sf(s, dfn=dfn, dfd=dfd)

    # this is the F-test for excessive variance
    dfss['anova'] = -np.log(dfss.apply(ftest, axis=1))

    df_ret = pd.DataFrame(dfss.anova).reset_index() \
        .merge(df0[['corpus', 'doc_tested', 'value']],
               on=['doc_tested', 'corpus'])
    return df_ret


def _comp_probs_t(df: pd.DataFrame, by: List, log_scale=True) -> pd.DataFrame:
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
    grp0 = df0.groupby(['corpus', 'author'])
    df_summary = grp0.agg({'value': ['mean', 'std', 'median', 'count']}, as_index=False).reset_index()
    print("======= Average value of within and between corpus discrepancies =======")
    print(df_summary[df_summary[('value', 'count')] > 1])

    df1 = df[df.kind != 'generic']

    value = 'value'

    df0 = df1[df1.doc_id == df1.doc_tested]
    df1 = df1.drop(df0.index)  # remove generic tests

    grp = df1.groupby(by)

    res = grp.agg({value: ['mean', 'std', 'median', 'mad', 'count']},
                  as_index=False).reset_index()

    res.loc[:, 'doc_id'] = res['doc_tested']

    dfm = df0.merge(res[['doc_id', 'corpus', 'value']],
                    on=['doc_id', 'corpus'], how='left')  # only include
    # results for generic chapters

    mu = dfm[(value, 'mean')]
    s = dfm[(value, 'std')]  # note that there is no need to adjust
    # for the number of DoF by multiplying by np.sqrt(n / (n - 1))
    # because the 'std' function of pandas uses (n-1) in denominator
    # when computing the std
    n = dfm[(value, 'count')]

    dfm.loc[:, 'rank_pval'] = 1 - dfm['rank']
    dfm.loc[:, 't-score'] = (dfm[value] - mu) / (s * np.sqrt(1 + 1 / n))
    if log_scale:
        dfm.loc[:, 't_test'] = -np.log(scipy.stats.t.sf(dfm['t-score'], df=n - 1))
    else:
        dfm.loc[:, 't_test'] = scipy.stats.t.sf(dfm['t-score'], df=n - 1)
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

    if params_report['anova']:
        dfm = _comp_probs_anova(df)
    else:
        dfm = _comp_probs_t(df, by=['author', 'doc_tested',
                                    'corpus'],
                                    log_scale=params_report['log_scale'])
    return dfm


def report_probs(dfm, params_report):
    """
    Arrange dfm as an easy-to-read table

    """

    value = params_report['value']
    dfm = dfm.rename(columns={'variable': value})

    if params_report['anova']:
        test_value_name = 'anova'
    else:
        test_value_name = 't_test'

    return dfm.pivot('corpus', 'doc_tested',
                     [value, test_value_name]
                     ).reset_index()


def report_probs_t(dfm, params_report):
    """
    Arrange dfm as an easy-to-read table 

    """
    value = params_report['value']
    dfm = dfm.rename(columns={'value': value})
    return dfm.pivot('corpus', 'doc_tested',
                     [value, 'rank_pval', 't_pval', 't-score']).reset_index()


def max_value_cls_accuracy(dfr, value):
    idx_score = dfr.groupby('doc_tested')[value].idxmin()
    return (dfr.loc[idx_score, 'corpus'] == dfr.loc[idx_score, 'author']).mean()


def max_value_cls_miss(dfr, value):
    idx_score = dfr.groupby('doc_tested')[value].idxmin()
    dfr_res = dfr.loc[idx_score]
    dfr_miss = dfr_res[dfr_res['corpus'] != dfr_res['author']]
    return dfr_miss.filter(['doc_id', 'corpus', value])


def summarize_probs_BS(dfm, params):
    """
    Evaluates accuracy and other stats for
     each BS iteration
    """

    grp = dfm.groupby('itr_BS')
    res = pd.DataFrame()
    for c in grp:
        r = summarize_probs(c[1], params)
        res = res.append(r, ignore_index=True)
    print(res)
    return res


def summarize_probs(dfm, params):
    """
    Print summary from probabilities evaluated in comp_probs
    This function is mostly for debugging purposes.
    """

    if params['anova']:
        test_value_name = 'anova'
    else:
        test_value_name = 't_test'
    value = params['value']

    if params['log_scale']:
        dfm.loc[:, 'sig_pval'] = np.exp(-dfm[test_value_name]) < params['sig_level']
    else:
        dfm.loc[:, 'sig_pval'] = dfm[test_value_name] < params['sig_level']
    print("========================RESULTS================================")

    dfm.loc[:, 'author'] = dfm.doc_tested.apply(_get_author_from_doc_id)
    dfr = dfm[dfm.author.isin(params['known_authors'])]

    acc_val = max_value_cls_accuracy(dfr, 'value')
    acc_test = max_value_cls_accuracy(dfr, test_value_name)
    miss_cls = max_value_cls_miss(dfr, test_value_name)

    print(f"Accuracy with {value}: ", acc_val)
    print(f"Accuracy with {test_value_name} of {value}: ", acc_test)
    print(f"Misclassified with {test_value_name}:")
    print(miss_cls)

    print(f"False Alarms with {test_value_name}:")
    print(dfr[(dfr.author == dfr.corpus) & dfr.sig_pval])

    df_res = dfr[dfr.author == dfr.corpus] \
        .groupby(['variable']) \
        .mean() \
        .filter(['sig_pval']) \
        .rename(columns={'sig_pval': f'FNR-{test_value_name}'})
    print(f"False alarm rate over {len(dfr)} chapters of known authorship:")
    print(df_res)

    return {'acc_value': acc_val,
            'acc_test': acc_test,
            #'miss_classified': miss_cls,
            'FNR': df_res.values[0]
            }


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
    print(res_tbl.loc['succ', :])
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


def report_table_known(df, report_params):
    """
    Arrange discrepancies test results to indicate accuracy of 
    authorship attribution of known authors

    Args:
    -----
    df                  discrepancies of many (doc, cropus) pairs
    report_params       parameters indicating how to report 

    """

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
    print(f"Reporting over chapters:\n{df_res.columns.tolist()}\n")
    print(f"Overall {df_res.shape[1] - 1} chapters:")
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
    res = grp.agg({'succ': ['mean','std','count']}, as_index=False).reset_index()

    print(res)
    return res


def _arrange_sim_full_results(res):
    """
    For HC, use max(HC, min_value)
    """
    min_value_hc = -5.0
    idcs = res.variable.str.contains(':HC')
    res.loc[idcs, 'value'] = np.maximum(res.loc[idcs, 'value'], min_value_hc)
    return res


def report_table_full_known(probs, params):
    """
    Params:
        :probs:     the output of compute_probs
        :params:    reporting parameters

    """

    dfr = probs[probs.author.isin(params['known_authors'])]
    value = params['value']

    if params['anova']:
        test_value_name = 'anova'
    else:
        test_value_name = 't_test'

    assert len(dfr.variable.unique()) == 1, "One and only one reportable variable is permitted"

    dfr = dfr.rename(columns={'value': value})
    dfm = dfr.pivot('corpus', 'doc_tested', [value, test_value_name])

    idx_min = dfm.idxmin()
    succ_val = idx_min[value].index.str.extract(r"([^|]+)")[0] == idx_min[value].values.tolist()
    succ_test = idx_min[test_value_name].index.str.extract(r"([^|]+)")[0] == idx_min[test_value_name].values.tolist()

    dfm.loc['succ', :] = pd.concat([succ_val, succ_test], ignore_index=True).values

    return dfm


def report_table_full_unknown(probs, params):
    dfr = probs[probs.author.isin(params['unknown_authors'])]
    value = params['value']

    assert len(dfr.variable.unique()) == 1, "One and only one reportable variable is permitted"

    dfr = dfr.rename(columns={'value': value})

    if params['anova']:
        test_value_name = 'anova'
    else:
        test_value_name = 't_test'

    dfm = dfr.pivot('corpus', 'doc_tested', [value, test_value_name])

    idx_min = dfm.idxmin()
    succ_val = idx_min[value].index.str.extract(r"([^|]+)")[0] == idx_min[value].values.tolist()
    succ_test = idx_min[test_value_name].index.str.extract(r"([^|]+)")[0] == idx_min[test_value_name].values.tolist()

    dfm.loc['succ', :] = pd.concat([succ_val, succ_test], ignore_index=True).values

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
