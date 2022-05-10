#pipeline: Data Engineering 

from kedro.pipeline import node, Pipeline
from .nodes import (process_data, add_topics,
                    add_convert, build_vocab,
                    add_to_report, merge_unknown)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=process_data, 
             inputs=["oshb_parsed", "params:preprocessing"],
             outputs="data_proc0",
            ),
        node(func=add_convert,
             inputs=["data_proc0", "oshb_parsed"],
             outputs="data_proc1",
            ),
        node(func=add_to_report,
             inputs=['data_proc1', 'reference_data'],
             outputs="data_proc2"
             ),
        node(func=merge_unknown,
             inputs=["data_proc2", "params:unknown_authors"],
             outputs="data_proc",
             ),
        node(func=build_vocab, 
             inputs=["data_proc", "params:vocab"],
             outputs="vocabulary1",
            ),
        node(func=add_convert,
             inputs=["vocabulary1", "oshb_parsed"],
             outputs="vocabulary",
            ),
        ], tags='data engineering'
    )