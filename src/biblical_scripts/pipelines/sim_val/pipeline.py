#pipeline: data science val 

"""
Same as sim_only but allows for custom cross validation procedure, obtained by
passing `k_fold` in `params:model`

"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.sim.nodes import (
     evaluate_accuracy, filter_by_author)
from biblical_scripts.pipelines.reporting.nodes import  (
     report_table_known, report_table_unknown)

from biblical_scripts.pipelines.plotting.nodes import plot_sim
from .nodes import (cross_validation)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author,
             inputs=["data_proc", "params:known_authors",
                     "params:unknown_authors",
                     "params:only_reportables"],
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
             inputs=["sim_res_cv", "params:report", "reference_data"],
             outputs="",
             name="plot_sim"
            )
        ], tags="Doc Similarity: cross validation"
    )
