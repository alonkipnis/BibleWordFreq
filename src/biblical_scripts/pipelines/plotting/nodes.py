# pipeline: plotting
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from plotnine import *
import plotnine

from biblical_scripts.pipelines.report.nodes import _prepare_res

plotnine.options.figure_size = (7, 9)

LIST_OF_COLORS = ['tab:red', 'tab:blue','tab:gray', "#00BA38", 
    'tab:olive', "#619CFF", 'tab:orange', "#F8766D",
    'tab:purple', 'tab:brown', 'tab:pink',
    'tab:green', 'tab:cyan', 'royalblue', 'darksaltgray', 'forestgreen',
    'cyan', 'navy'
    'magenta', '#595959', 'lightseagreen', 'orangered', 'crimson'
]

def _descibe_data(data, filename) :
    
    p = (ggplot(aes(x = 'author', fill='author'), data=data ) + geom_bar(stat='count')
    + ylab('# of lemmas') + coord_flip()
    )
    p.save(filename)


def _plot_author_pair(df, value, wrt_authors = [], show_legend=True):

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

def plot_sim(sim_full_res, params, known_authors) :
    """
    To do: create a partioned dataset for saving figs to disk
    """

    path = params['fig_path']
    value = params['value']
    
    df = _prepare_res(sim_full_res)
    
    df = df[df.len >= params['min_length_to_report']]
    #df['wrt_author'] = df.loc[:,'variable'].str.extract(r'([^:]+):') # get corpus name
    df_figs = pd.DataFrame()
    for auth1 in known_authors :
        for auth2 in known_authors :
            if auth1 < auth2 :
                auth_pair = (auth1, auth2)
                df_disp = df[df.author.isin(auth_pair) & df['variable'].str.contains(value)]
                fn = f'{auth1}_vs_{auth2}.png'
                p = _plot_author_pair(df_disp, value = 'value', wrt_authors=auth_pair) #+ xlim(0,15) + ylim(0,15)
                p.save(path + '/' + fn)
                #df_figs = df_figs.append({'authors' : auth_pair, 'fig' : p})
    return df_figs


def _add_prob(res1) :
    """
    Probability of observing score or more extreme
    """
    grp = res1.groupby(['doc','corpus'])
    res1['total'] = grp.variable.transform(pd.Series.count)
    res1['rank'] = pd.Series.round(grp.value.rank()-.5)
    res1['prob'] = 1 - res1['rank'] / res1['total']
    res1['prob'] = res1['prob'].round(3)
    return res1


def _plot_sim_full_doc(res1) :

    res1_full = res1[res1.author == 'doc_smp'] # only artificially gen'd data
    
    res1_doc = res1[res1.author == 'doc0'] # actual tested doc

    p = (ggplot(aes(x='value', fill = 'corpus', y='..density..',
                color='corpus', label='corpus'), data = res1_full)
             + geom_histogram(alpha=0.5,position='dodge') 
             #+ geom_density(alpha = 0.5)
             + geom_vline(aes(xintercept='value', color='corpus'),
                          data=res1_doc, size=1, linetype='dashed')
             + geom_label(data=res1_doc,
                              mapping=aes(x='value', y=0.5, label='prob', fill='corpus'),
                              position=position_jitter(),
                              size=10, colour = "black") ) 
    return p
    
def plot_sim_full(sim_full_res, params) :
    path = params['fig_path']
    
    res = _add_prob(sim_full_res)
    value = params['value']
    res = res[res.variable.str.contains(value)]
    lo_docs = res.doc.unique()
    res = res[~((res.doc_id == res.doc) & (res.author != 'doc0'))] # include only 
    # data obtained from testing doc_nm, but remove the record associated with 
    # doc_nm if one happended to be used in sampling
    
    for doc_nm in lo_docs :
        p = _plot_sim_full_doc(res[(res.doc == doc_nm)])
        p = p + ggtitle(f'{doc_nm}')
        p.save(path + '/' + doc_nm + '.png')


def plot_sim_full_BS(sim_full_res_BS, params) :
    """
    Illustrate results of BS
    
    To do:
     - add accuracy per BS iteration
     - Use BS std and std of accuracy when illustrating the results
    
    Right now we use the mean HC value which is meaningless
    
    """

    path = params['fig_path']
    
    res_BS = res_BS.rename(columns = {'value_mean' : 'value'}) # change things from here on
    
    res = _add_prob(res_BS)
    value = params['value']
    res = res[res.variable.str.contains(value)]
    lo_docs = res.doc.unique()
    res = res[~((res.doc_id == res.doc) & (res.author != 'doc0'))] # include only
    # data obtained from testing doc_nm, but remove the record associated with
    # doc_nm if one happended to be used in sampling
    
    for doc_nm in lo_docs :
        p = _plot_sim_full_doc(res[(res.doc == doc_nm)])
        p = p + ggtitle(f'{doc_nm} BS')
        p.save(path + '/' + doc_nm +'_BS'+'.png')


def plot_sim_BS(sim_full_res_BS, params, known_authors) :
    """
    To do: create a partioned dataset for saving figs to disk
    """
    
    path = params['fig_path']
    value = params['value']
    
    res = sim_full_res_BS[sim_full_res_BS.len >= params['min_length_to_report']]
    res = res.rename(columns = {'value_mean' : 'value'})
    
    df = res[res.author == 'doc0'].drop('author', axis=1)
    df = df.rename(columns = {'corpus' : 'wrt_author', 'true_author' : 'author'})
    
    df['wrt_author'] = df.loc[:,'variable'].str.extract(r'([^:]+):') # get corpus name
    df_figs = pd.DataFrame()
    for auth1 in known_authors :
        for auth2 in known_authors :
            if auth1 < auth2 :
                auth_pair = (auth1, auth2)
                df_disp = df[df.author.isin(auth_pair) & df['variable'].str.contains(value)]
                fn = f'{auth1}_vs_{auth2}_BS.png'
                p = _plot_author_pair(df_disp, value = 'value', wrt_authors=auth_pair) #+ xlim(0,15) + ylim(0,15)
                p.save(path + '/' + fn)
                #df_figs = df_figs.append({'authors' : auth_pair, 'fig' : p})
    return df_figs
