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
Make sure that you have [OSHB](https://github.com/openscriptures/morphhb) located at `data/01_raw/morphhb` and the table `data/01_raw/reference_data.csv` indicateing the parts from the bible we use. You can change the default locations of these files by modifying the data catalog `conf/base/catalog.yaml` (place your modified version on `conf/local/catalog.yaml`)

### Distributed bootstrap evaluation:
If you want to run the `sim_bs` pipeline using a distributed scheduler, install `dask-distributed` using 
```
python -m pip install dask distributed --upgrade
```

## Tunning pipelines:
The output of the pipelines are stored in the folder `data/` with sub-folders depending on each pipeline.  

### Specific pipeline:
```
kedro run --pipeline=pipeline_name
```
available pipelines:
 - `oshb`       read project data from OSHB Project data and arrange as a list of words, with verse, chapter, and author names. 
 - `de`     clean and perpare OSHB Project data
 - `sim`     compute Higher-Criticism (HC) similarity in a leave-one-out fashion.
 - `plot`       generate plots of results
 - `report`     generate report tables
 - `sim_bs`    bootstrap over lemmas and compute many instances of Higher-Criticism (HC) similarity in a leave-one-out fashion
- `plot_bs`     generate plots of results with bootstraped data

### All standard pipelines (`oshb`+`de`+`sim`+`plot`+`report`):
```
kedro run --pipeline=all
```
### The default pipelines are  (`de`+`sim`+`plot`+`report`), which you can get by running:
```
kedro run
```


