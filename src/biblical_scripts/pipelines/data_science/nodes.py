# pipeline: data science
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from biblical_scripts.pipelines.data_science.AuthorshipAttribution.AuthAttLib import AuthorshipAttributionDTM
from biblical_scripts.pipelines.data_science.AuthorshipAttribution.MultiDoc import CompareDocs
from typing import Dict, List
from plotnine import *
import plotnine
LIST_OF_COLORS = ['tab:red', 'tab:blue','tab:gray', "#00BA38", 
    'tab:olive', "#619CFF", 'tab:orange', "#F8766D",
    'tab:purple', 'tab:brown', 'tab:pink',
    'tab:green', 'tab:cyan', 'royalblue', 'darksaltgray', 'forestgreen',
    'cyan', 'navy'
    'magenta', '#595959', 'lightseagreen', 'orangered', 'crimson'
]

def _n_most_frequent_by_author(ds, n) :
    return ds.groupby(['author', 'feature'])\
            .count()\
            .reset_index()\
            .groupby(['author'])\
            .head(n).filter(['feature'])

def _n_most_frequent(ds, n) :
    return ds.groupby(['feature'])\
            .count()\
            .reset_index()\
            .head(n).filter(['feature'])

def build_vocab(data, params, known_authors) :
    n = params['no_tokens']
    by_author = params['by_author']
    
    ds = data[data.author.isin(known_authors)]
    if by_author :
        r = _n_most_frequent_by_author(ds, n)
    else :
        r = _n_most_frequent(ds, n)
    logging.info(f"Constructed a vocabulary of {len(r)} features")
    return r

def compute_sim(data : pd.DataFrame, vocabulary : pd.DataFrame, params, known_authors : List) -> pd.DataFrame :
    """
    Build model for comparing docs and authors based on word-frequencies
    
    Args:
    data        a dataframe representing tokens by docs by corpus
    vocabulary  a list of words by which docs are compared
    known_authors   only compare against corpora from this list
    
    Returns:
    df_res      Each row is the comparison of a doc against a corpus in known_authors
    """
    
    ds = data.rename(columns = {'feature' : 'term', 'chapter' : 'doc_id'}).dropna()
    ds['len'] = ds.groupby('doc_id').transform(pd.Series.count).iloc[:,0]
    par = {}
    [par.update(p) for p in params]
    ds = data.rename(columns = {'feature' : 'term', 'chapter' : 'doc_id'}).dropna()
    model = AuthorshipAttributionDTM(ds, **par, vocab=list(vocabulary.feature.values))
    
    # adding doc-length info
    doc_lengths = pd.DataFrame(ds.groupby('doc_id').count().rename(columns={'term' : 'len'}).len)
    df_res = model.compute_inter_similarity(LOO = True, wrt_authors=known_authors)
    df_res = df_res.merge(doc_lengths, on = 'doc_id')
    return df_res

def evaluate_accuracy(df_res: pd.DataFrame, known_authors : List, min_length_to_report : int, params) -> pd.DataFrame :
    """
    Evaluate accuracy of attribution in min discrepancy manner
    
    Args:
    min_length_to_report    only try to attribute docs of length larger than that
    known_authors           only compare against corpora from this list
    
    Returns:
    average accuracy for each similarity measure
    """
    def prob_of_succ(df, value, known_authors, min_length=min_length_to_report, plot=False) :
        df = df[df.len >= min_length]
        df.loc[:,value] = df[value].astype(float)
        res1 = df[df.author.isin(known_authors) & df.wrt_author.isin(known_authors)].reset_index()
        idx_min = res1.groupby(['doc_id', 'author'])[value].idxmin()
        res_min = res1.loc[idx_min, :]
        res_min.loc[:, 'succ'] = res_min.author == res_min.wrt_author

        succ_per_doc = res_min.groupby('doc_id').succ.mean()
        return succ_per_doc.mean()

    res = {}
    for v in ['HC', 'HC_rank', 'chisq', 'chisq_rank', 'log-likelihood','log-likelihood_rank', 'Fisher']  :
        res[v] = prob_of_succ(df_res, value = v, known_authors=known_authors)
        
    res['params'] = str(params)
    return pd.DataFrame(res, index=[0])

def _plot_author_pair(df, value = 'HC', wrt_authors = [],
                 show_legend=True):

    df.loc[:,value] = df[value].astype(float)
    df1 = df.filter(['doc_id', 'author', 'wrt_author', value])\
            .pivot_table(index = ['doc_id','author'],
                         columns = 'wrt_author',
                         values = [value])[value].reset_index()

    lo_authors = pd.unique(df.wrt_author)
    no_authors = len(lo_authors)

    if no_authors < 2 :
        raise ValueError

    if wrt_authors == [] :
        wrt_authors = (lo_authors[0],lo_authors[1])

    color_map = LIST_OF_COLORS

    df1.loc[:, 'x'] = df1.loc[:, wrt_authors[0]].astype('float')
    df1.loc[:, 'y'] = df1.loc[:, wrt_authors[1]].astype('float')
    p = (
        ggplot(aes(x='x', y='y', color='author', shape = 'author'), data=df1) +
        geom_point(show_legend=show_legend, size = 3) + geom_abline(alpha=0.5) +
        # geom_text(aes(label = 'doc_id', check_overlap = True)) +
        xlab(wrt_authors[0]) + ylab(wrt_authors[1]) +
        scale_color_manual(values=color_map) +  #+ xlim(0,35) + ylim(0,35)
        theme(legend_title=element_blank(), legend_position='top'))
    return p

def illustrate_results(df, value, known_authors) :
    """
    To do: create a partioned dataset for saving figs to disk
    """

    plotnine.options.figure_size = (7, 6)
    path = "data/08_reporting/Figs"

    df_figs = pd.DataFrame()
    for auth1 in known_authors :
        for auth2 in known_authors :
            if auth1 < auth2 :
                auth_pair = (auth1, auth2)
                df_disp = df
                df_disp = df[df.author.isin(auth_pair)]
                fn = f'{auth1}_vs_{auth2}.png'
                p = _plot_author_pair(df_disp, value = value, wrt_authors=auth_pair) #+ xlim(0,15) + ylim(0,15)
                p.save(path + '/' + fn)
                #df_figs = df_figs.append({'authors' : auth_pair, 'fig' : p})
    return df_figs
