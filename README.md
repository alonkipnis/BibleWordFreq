# BiblicalScripts

## Overview

Word-frequency analysis of Hebrew biblical texts of authorship attribution. 
Supporting material and code for a research article. The code is organized as 
a Kedro project ([Kedro documentation](https://kedro.readthedocs.io)).

## How to install dependencies
Run:
```
kedro install
```
### Raw dataset:
Make sure that you have [OSHB](https://github.com/openscriptures/morphhb) 
located in `data/01_raw/morphhb` and the table `data/01_raw/reference_data.csv` 
indicateing the parts from the bible we use and their corrsponding authorship
information. You can change the default locations of these files by modifying
the data catalog `conf/base/catalog.yaml` (place your modified version on 
`conf/local/catalog.yaml`)

### Distributed bootstrap evaluation:
If you want to run the `sim_bs` pipeline with a distributed scheduler, install 
`dask-distributed` using 
```
python -m pip install dask distributed --upgrade
```

# General Kedro Instruction 

## Tunning pipelines:
The output of the pipelines are stored in the folder `data/` with sub-folders 
depending on each pipeline. At a high-level, `data/07_sim_output` contains tables
 with raw evaluations of Higher Critisim or other discrepancies measures, and
`data/08_reporting` contains figures and human-friendly reports summarizing 
raw evaluations. 


### Specific pipeline:
```
kedro run --pipeline=pipeline_name
```
available pipelines:
 - `oshb`    :    read project data from OSHB Project data and arrange as a list
  of words, with verse, chapter, and author names. 
 - `de`      :    clean and perpare OSHB Project data
 - `sim_only`:    Higher-Criticism (HC) discrepancy in a leave-one-out fashion 
                  for data of known authorship 
 - `sim_val` :    Same as `sim_only` but allows for custom cross-validation 
                  procedure by passing `k_fold` in `params:model`
 - `sim_full`:    allow the evaluation of propoability of doc - corpus 
                  association by "simulating" a more accurate null distribution
 - `plot`    :    generate plots from model evaluations
 - `report`  :    generate report tables
 - `sim_bs`  :    bootstrap over lemmas and compute many instances of Higher- 
                  Criticism (HC) discrepancy in a leave-one-out fashion
- `plot_bs`  :    generate plots of results for bootstraped data
- `chunk_len`:    assess accuracy of authorship attribution versus document 
                  length. This is acheived by randomly sampling a chunk from 
                  a corpus of homogenous authorship and attempting to associate
                  that chunk to the correct corpus. 

### All standard pipelines (`oshb`+`de`+`sim`+`plot`+`report`):
```
kedro run --pipeline=all
```
### The default pipelines sequence is (`de`+`sim`+`plot`+`report`). You can 
initiate this sequence simply by:
```
kedro run
```


