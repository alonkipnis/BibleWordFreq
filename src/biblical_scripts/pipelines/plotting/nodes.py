# pipeline: plotting
# project: bib-scripts
import pdb

import pandas as pd
import logging
from plotnine import *
import plotnine

# from biblical_scripts.pipelines.report.nodes import _prepare_res

plotnine.options.figure_size = (7, 9)

LIST_OF_COLORS = ['tab:red', 'tab:blue', 'tab:gray', "#00BA38",
                  'tab:olive', "#619CFF", 'tab:orange', "#F8766D",
                  'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:green', 'tab:cyan', 'royalblue', 'darksaltgray', 'forestgreen',
                  'cyan', 'navy'
                          'magenta', '#595959', 'lightseagreen', 'orangered', 'crimson'
                  ]


def _describe_data(data, filename):
    p = (ggplot(aes(x='author', fill='author'), data=data) + geom_bar(stat='count')
         + ylab('# of lemmas') + coord_flip()
         )
    p.save(filename)


def _plot_author_pair(df, value, wrt_authors=None, show_legend=True):
    assert (len(df) > 0), "Perhaps variable to report is not predicted by model?"
    df1 = df.astype({value: float}) \
        .filter(['doc_id', 'author', 'corpus', value]) \
        .pivot_table(index=['doc_id', 'author'],
                     columns='corpus',
                     values=[value])[value] \
        .reset_index()
    # arrange results so that rows corresponding to authors and columns to chapters

    lo_authors = pd.unique(df.corpus)
    no_authors = len(lo_authors)

    if no_authors < 2:
        raise ValueError

    if wrt_authors is None:
        wrt_authors = (lo_authors[0], lo_authors[1])

    color_map = LIST_OF_COLORS

    df1.loc[:, 'x'] = df1.loc[:, wrt_authors[0]].astype('float')
    df1.loc[:, 'y'] = df1.loc[:, wrt_authors[1]].astype('float')
    p = (
            ggplot(aes(x='x', y='y', color='author', shape='author'), data=df1) +
            geom_point(show_legend=show_legend, size=3) + geom_abline(alpha=0.5) +
            # geom_text(aes(label = 'doc_id', check_overlap = True)) + # show doc names
            xlab(wrt_authors[0]) + ylab(wrt_authors[1]) +
            scale_color_manual(values=color_map) +  # + xlim(0,35) + ylim(0,35)
            theme(legend_title=element_blank(), legend_position='top'))
    return p


def _arrange_metadata(df, value):
    """
    add 'corpus' and 'author' column to evaluation results
    by parsing 'variable' and 'doc_id' columns.
    """

    df = df[df.variable.str.contains(value)]
    df.loc[:, 'corpus'] = df.variable.str.extract(rf"(^[A-Za-z0-9 ]+)(-ext)?:([A-Za-z]+)")[0]
    df.loc[:, 'author'] = df.doc_id.str.extract(r"^([^|]+)|")[0]
    df.loc[:, 'variable'] = value
    return df


def plot_sim(sim_res, params, reference_data):
    """
    Illustrate discrepancy results

    To DO:
        create a partioned dataset for saving figs to disk
    """

    known_authors = params['known_authors']
    path = params['fig_path']
    value = params['value']

    to_report = reference_data[reference_data.to_report]
    lo_chapters_to_report = to_report['author'] + '|' + to_report['book'] + '.' + to_report['chapter'].astype(str)

    df = _arrange_metadata(sim_res[sim_res.doc_id.isin(lo_chapters_to_report)], value)
    # col names compatible with _plot_author_pair

    df = df[df.len >= params['min_length_to_report']]
    assert(len(df) > 0), "No data"
    # get corpus name
    figs = {}
    for auth1 in known_authors:
        for auth2 in known_authors:
            if auth1 < auth2:
                auth_pair = (auth1, auth2)
                df_disp = df[df.author.isin(auth_pair) & df['variable'].str.contains(value)]
                fn = f'{auth1}_vs_{auth2}.png'
                p = _plot_author_pair(df_disp, value='value', wrt_authors=auth_pair)  # + xlim(0,15) + ylim(0,15)
                figs[auth_pair] = p
                p.save(path + '/' + fn)
    return figs


def _add_prob(res1):
    """
    Rank test: 
    Probability of observing a score or more extreme
    based on given scores.
    """
    grp = res1.groupby(['doc_tested', 'corpus'])
    res1['total'] = grp.variable.transform(pd.Series.count)
    res1['rank'] = pd.Series.round(grp.value.rank() - .5)
    res1['prob'] = 1 - res1['rank'] / res1['total']
    res1['prob'] = res1['prob'].round(3)
    return res1


def _plot_sim_full_doc(res1):
    """
    Illustrate the empirical distribution of a document ()
    
    """
    res1_full = res1[res1.kind == 'ext']  # only artificially gen'd data
    res1_doc = res1[res1.kind == 'org']  # actual tested doc

    p = (ggplot(aes(x='value', fill='corpus', y='..density..',
                    color='corpus', label='corpus'), data=res1_full)
         + geom_histogram(alpha=0.5, position='dodge')
         # + geom_density(alpha = 0.5)
         + geom_vline(aes(xintercept='value', color='corpus'),
                      data=res1_doc, size=1, linetype='dashed')
         + geom_label(data=res1_doc,
                      mapping=aes(x='value', y=0.5, label='prob', fill='corpus'),
                      position=position_jitter(),
                      size=10, colour="black"))
    return p


def plot_sim_full(sim_full_res, params):
    path = params['fig_path']
    value = params['value']
    res = _add_prob(_arrange_metadata(sim_full_res, value))

    res = res[res.variable.str.contains(value)]
    lo_docs = res.doc_tested.unique()
    res = res[~((res.doc_id == res.doc_tested) & (res.kind == 'ext'))]  # include only
    # data obtained from testing doc_nm, but remove the record associated with 
    # doc_nm if one happended to be used in sampling

    for doc_nm in lo_docs:
        p = _plot_sim_full_doc(res[(res.doc_tested == doc_nm)])
        p = p + ggtitle(f'{doc_nm}')
        try:
            p.save(path + '/' + doc_nm + '.png')
        except:
            logging.error(f"Could not save {doc_nm}.png")


def plot_sim_full_bs(sim_full_res_bs, params):
    """
    Illustrate results of Bootstrap evaluations
    
    To do:
     - add accuracy per BS iteration
     - Use BS std and std of accuracy when illustrating the results
    
    Right now we use the mean HC value which is meaningless
    
    """

    path = params['fig_path']
    value = params['value']

    res_bs = sim_full_res_bs.rename(columns={'value_mean': 'variable'})
    # change things from here on

    res = _add_prob(_arrange_metadata(res_bs, value))

    value = params['value']
    res = res[res.variable.str.contains(value)]
    lo_docs = res.doc.unique()
    res = res[~((res.doc_id == res.doc) & (res.author != 'doc0'))]  # include only
    # data obtained from testing doc_nm, but remove the record associated with
    # doc_nm if one happened to be used in sampling

    for doc_nm in lo_docs:
        p = _plot_sim_full_doc(res[(res.doc == doc_nm)])
        p = p + ggtitle(f'{doc_nm} BS')
        p.save(path + '/' + doc_nm + '_BS' + '.png')

