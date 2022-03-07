"""
This is a boilerplate pipeline 'features'
generated using Kedro 0.17.0
"""

import pandas as pd
import numpy as np
from biblical_scripts.extras.Convert import Convert
from biblical_scripts.pipelines.sim.nodes import build_model
import logging

from bidi import algorithm as bidialg
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def _remove_backslash(text):
    """
    removes backslash prefix

    Since hebrew is right to left, the prefix is to prefix is to the right
    """
    return text.split("/")[0]

def _remove_teamim(my_string):
    """
    Remove teamim from hebrew text
    """
    return ''.join(['' if (ord(c) <= 1456) and (ord(c) > 123) else c for c in my_string])

def _replace_special(text):
    """
    Replace special lemma codes with their names
    """
    if text == "<Np>":
        return "(proper name)"
    elif text == "<Ng>":
        return "(gentilic name)"
    elif text == "<Ac>":
        return "(cardinal number)"
    elif text == "<Pp>":
        return "(personal proposition)"
    elif text == "<Nc>":
        return "(common noun)"
    elif text == "<Vq>":
        return "(verb)"
    return text


def _prepare_text(text):
    return _remove_teamim(
        _remove_backslash(
            _replace_special(text)))

def _plot_words_volcano(dfa):
    """
    Volcano plot of P-values
    """

    k = len(dfa)
    L = int(k / 3)
    fig = figure(figsize=(4, L), dpi=80)
    dfa.loc[:, 'value'] = -np.log(dfa.pval) * dfa['affinity']

    dfb = dfa[::-1]

    ax1 = plt.axes()

    tab_blue = '#288FB7'
    tab_gray = '#7F7F7F'
    tab_brown = "#8C564B"
    light_gray = "gray"

    plt.barh(range(k), dfb['value'], color=light_gray,
             height=1, alpha=.5)

    for i, r in enumerate(dfb.iterrows()):
        # Create an axis text object
        sgn = 2 * (r[1]['value'] > 0) - 1

        wl = len(r[1].term)
        x = r[1]['value'] / 2
        text = _prepare_text(f'{r[1].term}')
        plt.text(x,  # X location of text (with adjustment)
                 i,  # Y location
                 s=bidialg.get_display(text),  # Required label with formatting
                 va='center',  # Vertical alignment
                 ha='center',  # Horizontal alignment
                 color='black',  # Font colour and size
                 # backgroundcolor='white',
                 fontsize=12, name="Ariel")
        plt.xlabel('log(p)')

    # ax1.get_yaxis().set_visible(True)
    plt.yticks([])
    plt.xticks([-100, -50, 0, 50, 100])
    return plt.gca()


def get_features(data, vocab, data_raw, model_params, params_features):

    lo_authors = params_features['known_authors']
    lo_chapters = params_features['specific_chapters']

    md = build_model(data[data.author.isin(lo_authors) | data.chapter.isin(lo_chapters)],
                     vocab, model_params)
    df = md[0].HCT_vs_many()

    for auth in lo_authors:
        df.loc[:, f'{auth}:freq'] = df[f'{auth}:n'] / df[f'{auth}:T']

    df.loc[:, "freq_common"] = df['n'] / df['T']

    dfm = df[df.iloc[:,  # only use features selected at least once
             df.columns.str.contains('affinity')].abs().any(axis=1)].reset_index()

    cvr = Convert(data_raw)
    dfm['term'] = dfm['feature'].apply(cvr._lem2term)
    return dfm


def get_features_chapter(data, vocab, data_raw, model_params, params_features):
    lo_authors = params_features['known_authors']
    lo_chapters = params_features['specific_chapters']

    for ch in lo_chapters:
        for auth in lo_authors:
            logging.info(f"Checking features: {ch} vs. {auth}")
            ds = data[(data.author == auth) | data.chapter.str.contains(fr"{ch}")]
             # WARNING: here we may have a match if chapter string
             # equals the author of ch
            ds.loc[data.chapter.str.contains(fr"{ch}"), 'author'] = ch


            md = build_model(ds, vocab, model_params)
            df = md[0].HCT_vs_many()
            df.loc[:, f'{auth}:freq'] = df[f'{auth}:n'] / df[f'{auth}:T']
            df.loc[:, f'{ch}:freq'] = df[f'{ch}:n'] / df[f'{ch}:T']
            df.loc[:, "freq_common"] = df['n'] / df['T']
            dfm = df[df.iloc[:,  # only use features selected at least once
                     df.columns.str.contains('affinity')].abs().any(axis=1)].reset_index()

            cvr = Convert(data_raw)
            dfm['term'] = dfm['feature'].apply(cvr._lem2term)
            dfm.to_csv(f"{params_features['out_path']}{ch}_vs_{auth}.csv")


def plot_features(dfm, params_features):
    """
    Plot discriminating features for each author in `lo_auth`

    Args:
        `k`   is the total number of features to display
    """

    fig_path = params_features['fig_path']
    k = params_features['num_features_to_plot']
    lo_auth = params_features['known_authors']

    for auth in lo_auth:
        dfa = dfm.reset_index() \
            .rename(columns={f'{auth}:pval': 'pval', f'{auth}:affinity': 'affinity', f'{auth}:freq': 'freq'}) \
            .filter(['term', 'feature', 'freq', 'pval', 'affinity', 'freq_common'])
        dfa = dfa.sort_values('pval', ascending=True)
        dfa = dfa[dfa['affinity'] != 0]
        #dfa.to_csv(f"/Users/kipnisal/DS/BiblicalScripts/bib-scripts/data/08_reporting/features_{auth}.csv")
        chk = np.mean(2 * (dfa['freq'] > dfa['freq_common']) - 1 == dfa['affinity'])
        assert np.abs(chk - 1) < 1e-4, "Affinity check failed!"

        _plot_words_volcano(dfa[:k])
        plt.title(f"Top {k} Discriminating Words for {auth}")
        plt.savefig(f"{fig_path}/feature_{auth}.png")


