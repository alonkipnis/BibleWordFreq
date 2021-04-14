#pipeline: reporting

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science_val.nodes import (cross_validation)
from biblical_scripts.pipelines.report.nodes import report_sim

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=report_sim,
             inputs=["sim_null_res", "vocabulary", "params:model", "params:report"],
             outputs="sim_report"
            ),
        ]
    )
