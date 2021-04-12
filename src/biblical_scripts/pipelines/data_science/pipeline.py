#pipeline: data science

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science.nodes import (compute_sim, evaluate_accuracy)
from biblical_scripts.pipelines.plotting.nodes import illustrate_results

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=compute_sim, 
             inputs=["data_proc", "vocabulary", "params:model", "params:known_authors"],
             outputs="sim_res",
             name="compute_HC_similarity",
            ), 
        node(func=evaluate_accuracy, 
             inputs=["sim_res", "params:known_authors", "params:report", "parameters"],
             outputs="sim_acc",
             name="evaluate_accuracy",
            ),
        node(func=illustrate_results,
             inputs=["sim_res", "params:report", "params:known_authors"],
             outputs="",
             name="illustrate"
            )
        ]
    )