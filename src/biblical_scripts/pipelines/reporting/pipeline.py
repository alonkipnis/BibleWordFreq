#pipeline: reporting

from kedro.pipeline import node, Pipeline
from .nodes import (report_sim_full, report_table_full_known,
          report_table_full_unknown, comp_probs, report_probs)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=report_table_full_known,
             inputs=["sim_full_res", "params:report",
                     "params:known_authors", "chapters_to_report"],
             outputs="report_table_full_known"
            ),
        node(func=report_table_full_unknown,
             inputs=["sim_full_res", "params:report",
                     "params:unk_authors"],
             outputs="report_table_full_unk"
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
