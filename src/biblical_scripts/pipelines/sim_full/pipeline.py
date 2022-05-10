# pipeline: sim full

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
a P-value under the model that documents are sampled independently from
a corpus. The functions 
`report_table_full_known` 
report_table_full_unknown`
Report on the P-value from this rank-based test, indicate the corpus
attaining maximal P-value (maximum likelihood under the null), and
indicate whether we cannot reject the null hypothesis at level .05

Another way to obtain probabilities and P-values is to 
suppose that HC discrepancies between documents and corporas follow
a normal distribution, so that we are looking at a t-test of the HC
scores with mean and std obtained by the sample mean and std, respectively. 
The functions `comp_probs` computes these probabilities, while the function
`report_probs` output these probabilities in a table that is readable. 

"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import (evaluate_accuracy,
                                                  filter_by_author)

from biblical_scripts.pipelines.reporting.nodes import (
    report_table_full_known, report_table_full_unknown,
    comp_probs, report_probs, summarize_probs)

from .nodes import sim_full


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=filter_by_author,
                 inputs=["data_proc", "params:all_authors",
                         "params:unknown_authors", "params:only_reportables"],
                 outputs="data",
                 name="filter_by_author"
                 ),
            node(func=sim_full,
                 inputs=["data", "params:vocab", "params:model",
                         "params:sim_full", "params:known_authors",
                         "reference_data"
                         ],
                 outputs="sim_full_res",
                 name="sim_full"
                 ),
            node(func=comp_probs,
                 inputs=["sim_full_res", "params:report"],
                 outputs="probs",
                 name="comp_probs"),
            node(func=report_probs,
                 inputs=["probs", "params:report"],
                 outputs="probs_table",
                 name="report_probs"
                 ),
            node(func=report_table_full_known,
                 inputs=['probs', 'params:report'],
                 outputs="sim_full_table_report_known",
                 name="report_table_full_known"
                 ),
            node(func=report_table_full_unknown,
                 inputs=['probs', 'params:report'],
                 outputs="sim_full_table_report_unknown",
                 name="report_table_full_unknown"
                 ),
            node(func=summarize_probs,
                 inputs=['probs', 'params:report'],
                 outputs="false_negative_rates",
                 name="summarize_probs"
                 )
        ], tags='minimal rank discrepancy'
    )
