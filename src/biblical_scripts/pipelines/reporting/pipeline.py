#pipeline: reporting

from kedro.pipeline import node, Pipeline
from .nodes import (report_sim_full, report_table_full,
           comp_probs, report_probs, )

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=report_sim_full,
             inputs=["sim_full_res", "params:report"],
             outputs="sim_full_report"
            ),
        node(func=report_table_full,
             inputs=["sim_full_res", "params:report"],
             outputs="sim_table_report"
            ),
        node(func=comp_probs,
             inputs=["sim_full_res", "params:report"],
             outputs="probs",
             name="comp_probs"
            ),
        node(func=report_probs,
             inputs=["probs", "params:report"],
             outputs="probs_table",
             name="report_probs"
            )
        ], tags="reporting"
    )
