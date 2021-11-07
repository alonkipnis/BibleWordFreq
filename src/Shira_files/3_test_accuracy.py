#!/usr/bin/env python
# coding: utf-8

# # Test Accuracy
# 

# In[1]:


import pandas as pd
import numpy as np
import re

#import auxiliary functions for python
import sys
sys.path.append('./')

from plotnine import *
from AuthAttLib import *
from visualize_HC_scores import *
from tqdm import tqdm


# ## Load and Arrange Data
# 
# #### Bible with morphology info
# OpenBible project inforation: <a href = 'https://hb.openscriptures.org/index.html'>https://hb.openscriptures.org/index.html</a> <br>
# Data is avaialbe at <a href='https://github.com/openscriptures/morphhb/tree/master/wlc'>https://github.com/openscriptures/morphhb/tree/master/wlc</a> <br>
# Information on morphological code is available at <a href='https://hb.openscriptures.org/parsing/HebrewMorphologyCodes.html'>https://hb.openscriptures.org/parsing/HebrewMorphologyCodes.html</a>
# 

# ### Options:
# - remove proper names (or other parts of speech)
# - count connecting words seperately
# - group ngrams
# - add 'topic' field to group multiple chapters together

# In[13]:


from load_biblical_scriptures import *

raw_data = read_biblical_scriptures(from_morphh=False)


NG_RANGE=(1,3)

data = process_data(raw_data,
    extract_prefix=True, # split lemmas by prefix and suffix
    extract_suffix=True,
    ng_range=NG_RANGE, # (min_ngram, max_ngram)
    #pad=True, # add <start> symbol at the beginning of a verse
    to_remove = [], # morpholical codes to remove (Np = proper name, Ng = gentilic noun), 
        #see https://hb.openscriptures.org/parsing/HebrewMorphologyCodes.html
    to_replace= ["Np","Ac", "Ng"],  # morpholical codes to replace"Nc", "Np" ,"Ng"
    add_topics=True,
    flat_suff_person=True)


# In[14]:


ds = data.copy()
ds = ds.filter(['chapter', 'lemma', 'author', 'morph', 'verse','term','topic'])          .rename(columns = {'chapter' : 'doc_id', 'lemma' : 'term'})

lo_unknown_authors = ds[ds.author.str.contains('UNK')].author.unique()
for auth in lo_unknown_authors :
    ds.loc[ds.author==auth,'doc_id'] = auth


# #### Most common words:

# In[15]:


N_WORDS = 500

def n_most_frequent_by_author(ds, n) :
    terms = ds.groupby(['author','term'])            .count()            .sort_values('doc_id', ascending=False)            .reset_index()            .groupby(['author'])            .head(n)

    return terms.term.unique().tolist()

most_freq = n_most_frequent_by_author(ds[ds.author.isin(['Dtr', 'DtrH', 'P'])],
                                      N_WORDS)
print("size of vocabulary = ", len(most_freq))


# # Test Accuracy by Sampling Random Chunks
# Current implementation samples contiguous chunks of 'chunk_size' verses

# In[16]:


# Test accuracy of sampled docs
MIN_CNT = 3
GAMMA = 0.25

def test_chunck_1(ds, author, chunk_size, sampling_method='verse') :
    
    
    ds1 = ds.copy()
    lo_authors = ds.author.unique()
    ds1 = ds1.set_index(sampling_method)
    pool = ds1[ds1.author == author].index.unique()

    k = np.random.randint(1, len(pool)-chunk_size) #sample contigous part
    # TODO: arrange in the right order (not alphabetically)
    k1 = np.random.randint(chunk_size, len(pool)) 
    smp = pool[k:k+chunk_size]#k1] #
    ds1.loc[smp, 'author'] = '<TEST>'
    ds1.loc[smp, 'doc_id'] = '<TEST>'
    ds1.author.unique()

    md = AuthorshipAttributionDTM(ds1.reset_index(), min_cnt = MIN_CNT,
                vocab=most_freq, gamma = GAMMA, verbose = False)
    #res = md.comp

    res = md.compute_inter_similarity(authors=['<TEST>'], wrt_authors=lo_authors)
    return res.dropna(axis=1)


# In[22]:


def test_accuray(ds, chunk_size, sampling_method='verse', value='HC', nMonte=10) :
    lo_known_authors = ds.author.unique()
    nMonte = 100
    #chunk_size = 30 

    res = pd.DataFrame()
    ds1 = ds[ds.author.isin(lo_known_authors)]

    for i in tqdm(range(nMonte)) :
        for auth in lo_known_authors :
            res1 = test_chunck_1(ds1, author = auth, chunk_size = chunk_size, sampling_method='verse')
            res1.loc[:,'test_id'] = auth + '-'+ str(i)
            res1.loc[:,'true_author'] = auth
            res = res.append(res1, ignore_index=True)

    tt = res.groupby('test_id')
    idx = tt[value].idxmin()
    acc = np.mean(res.loc[idx].wrt_author == res.loc[idx].true_author)
    return acc


# ### Accuracy for a single chunk size

# In[ ]:


#acc = test_accuray(ds[ds.author.isin(['Dtr', 'P', 'DtrH'])],
#                   chunk_size=30,
#                   nMonte=10)
#print("Accuracy = ", acc)


# ### Iterate over many chunk sizes

# In[ ]:


acc = []
for chunk_size in [10,100,200,400,450,500,700,900,1500,2000,3500] : #5, 10, 20, 30, 40, 50, 60, 80, 100
    acc += [test_accuray(ds[ds.author.isin(['Dtr', 'P', 'DtrH'])], chunk_size=chunk_size, nMonte=10)]
    print(chunk_size)
    print(acc)

print(acc)
#0.9333333333333333, 0.8666666666666667, 0.9333333333333333, 0.9666666666666667, 0.9333333333333333, 0.9, 0.9666666666666667, 0.9, 0.9]