# pipeline: plotting
# project: bib-scripts

import pandas as pd
import numpy as np
import logging
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

def _descibe_data(data, filename) :
    plotnine.options.figure_size = (7.4, 14)
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

def plot_sim(df, params, known_authors) :
    """
    To do: create a partioned dataset for saving figs to disk
    """

    plotnine.options.figure_size = (7, 6)
    path = "data/08_reporting/Figs"

    value = params['value']
    df = df[df.len >= params['min_length_to_report']]
    df['wrt_author'] = df.loc[:,'variable'].str.extract(r'([^:]+):') # get corpus name
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
