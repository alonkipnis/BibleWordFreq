#pipeline: data science val 

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science.nodes import (evaluate_accuracy, filter_by_author, report_table_known, report_table_unknown)

from biblical_scripts.pipelines.data_science_val.nodes import (cross_validation)
from biblical_scripts.pipelines.plotting.nodes import plot_sim

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:all_authors", "params:unk_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=cross_validation,
             inputs=["data", "vocabulary", "params:model"],
             outputs="sim_res_cv"
            ),
        node(func=report_table_known,
             inputs=["sim_res_cv", "params:report"],
             outputs="sim_table_report_known",
             name="report_table_known",
            ),
        node(func=report_table_unknown,
             inputs=["sim_res_cv", "params:report"],
             outputs="sim_table_report_unknown",
             name="report_table_unknown",
            ),
        node(func=plot_sim,
             inputs=["sim_res_cv", "params:report"],
             outputs="",
             name="illustrate"
            )
        ]
    )
