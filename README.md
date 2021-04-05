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

## How to run the pipelines:

### Specific pipeline:
```
kedro run --pipeline=pipeline_name
```
available pipelines:
 * 'oshb'   read project data from OSHB Project data and arrange as a list of
 			words, with verse, chapter, and author names. 
 * 'de'     clean and perpare OSHB Project data
 * 'ds'		Compute Higher-Criticism (HC) similarity in a leave-one-out fashion.

### All pipelines:
```
kedro run
```

