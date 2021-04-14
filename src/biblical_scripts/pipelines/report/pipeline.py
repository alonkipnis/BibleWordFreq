#pipeline: reporting

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science_val.nodes import (cross_validation)
from biblical_scripts.pipelines.report.nodes import (report_sim, report_table)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=report_sim,
             inputs=["sim_full_res", "vocabulary", "params:model", "params:report"],
             outputs="sim_report"
            ),
        node(func=report_table,
             inputs=["sim_full_res"],
             outputs="sim_table_report"
            ),
        ]
    )
