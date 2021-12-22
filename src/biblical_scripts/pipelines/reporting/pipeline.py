#pipeline: reporting

from kedro.pipeline import node, Pipeline
from .nodes import (comp_probs, report_probs, summarize_probs)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=comp_probs,
             inputs=["sim_full_res", "params:report"],
             outputs="probs",
             name="comp_probs"
            ),
        node(func=report_probs,
             inputs=["probs", "params:report"],
             outputs="probs_table",
             name="report_probs"
            ),
        node(func=summarize_probs,
             inputs=['probs', 'params:report', 'chapters_to_report'],
             outputs="false_negative_rates",
             name="summarize_probs"
             )
        ], tags="reporting"
    )
