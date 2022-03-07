"""
This is a boilerplate pipeline 'chunk_len'
generated using Kedro 0.17.0
"""

from kedro.pipeline import Pipeline, node

#from biblical_scripts.pipelines.reporting.nodes import  (
#     report_table_len)

from biblical_scripts.pipelines.sim.nodes import (
    filter_by_author)

from biblical_scripts.pipelines.reporting.nodes import (
    report_table_len)


from .nodes import (test_chunks_len)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author,
             inputs=["data_proc", "params:known_authors",
                     "params:unknown_authors",
                     "params:only_reportables"],
             outputs="data_filtered",
             name="filter_by_author"
            ),
        node(func=test_chunks_len,
             inputs=["data_filtered", "vocabulary", "params:known_authors"
              ,"params:model", "params:chunk_len_params"],
             outputs="sim_len_res",
             name="run_chunk_len"
            ),
        node(func=report_table_len,
             inputs=["sim_len_res", "params:report"],
             outputs="sim_len_table_report",
             name="report_table_len",
            ),
        ], tags='accuracy vs. doc length'
    )
