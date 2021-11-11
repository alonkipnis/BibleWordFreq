#pipeline: sim full

"""
The purpose of this pipeline is to allow the evaluation
of propoability of doc - corpus association.

The sim_full pipeline consideres a list of documents and a list in which
each item is a corpus of known authorship ('generic corpus').
For each document, it executes the following procedure:
1. Mark the document as 'TEST' corpus
2. For each generic corpus :
    2.1. Combine corpus and TEST corpus to form an 'extended corpus'
    2.2. For each doc in the 'extended corpus':
        2.2.1. Evalaute d(doc, extended corpus - doc) in 
        the standard way


We may treat all discrepancy scores obtained in step 2.2.1
as samples from the null hypothesis that the document and the corpus
belong to the same author. The rank of d(doc, generic corpus) with
respect to those samples can be used to associate or disassociate doc
with the author of the generic corpus. This rank can be converted to 
a P-value under the model that documents are sampled indepdently from 
a corpus

"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import (evaluate_accuracy, 
    filter_by_author)

from biblical_scripts.pipelines.reporting.nodes import (
    report_table_full_known, report_table_full_unknown)

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
            ),
        node(func=report_table_full_known,
            inputs=['sim_full_res', 'params:report', 'params:known_authors'],
            outputs="sim_full_table_report_known",
            name="report_table_full_known"
            ),
        node(func=report_table_full_unknown,
            inputs=['sim_full_res', 'params:report', 'params:unk_authors'],
            outputs="sim_full_table_report_unknown",
            name="report_table_full_unknown"
            )
        ], tags='minimal rank discrepancy'
    )
