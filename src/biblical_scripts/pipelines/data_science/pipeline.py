"""
Data Engineering Pipeline. 
"""

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science.nodes import (build_vocab, compute_sim, evaluate_accuracy, illustrate_results)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=build_vocab, 
             inputs=["data_proc", "params:vocab", "params:known_authors"],
             outputs="vocabulary",
             name="build_vocab"
            ),
            # to do: 
        node(func=compute_sim, 
             inputs=["data_proc", "vocabulary", "params:HC", "params:known_authors"],
             outputs="sim_res",
             name="compute_HC_similarity",
            ), 
        node(func=evaluate_accuracy, 
             inputs=["sim_res", "params:known_authors", "params:min_length_to_report", "parameters"],
             outputs="sim_acc",
             name="evaluate_accuracy",
            ),
        node(func=illustrate_results,
             inputs=["sim_res", "params:value", "params:known_authors"],
             outputs="",
             name="illustrate"
            )
        ]
    )