#pipeline: sim full

"""
The purpose of this pipeline is to allow the evaluation
of propoability of doc - corpus association.

The sim_full pipeline consideres a list of documents and a list in which
each item is a corpus of known authorship ('generic corpus').
For each document, it executes the following procedure:
1. Mark this document as 'TEST' corpus
2. For each author in knonw_authors :
    2.1. Combine corporas TEST and author to form an 'extended corpus'
    2.2. For each doc in the 'extended corpus':
        2.2.1. Evalaute d(doc, extended corpus) in the standard way


We can now treat all discrepancy scores obtained in step 2.2.1
as samples from the null hypothesis that the document and the corpus
belong to the same author. The rank of d(doc, generic corpus) with
resect to those samples.

"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import (evaluate_accuracy, 
    filter_by_author)

from .nodes import sim_full

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:all_authors", "params:unk_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=sim_full,
             inputs=["data", "vocabulary", "params:model",
                         "params:sim_full", "params:known_authors"],
             outputs="sim_full_res",
            name="sim_full"
            )
        ], tags='full similarity'
    )
