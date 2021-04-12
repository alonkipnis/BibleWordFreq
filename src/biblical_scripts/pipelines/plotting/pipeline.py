#pipeline: plotting

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.plotting.nodes import illustrate_results

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=illustrate_results,
             inputs=["sim_res", "params:report", "params:known_authors"],
             outputs="",
             name="illustrate"
            )
        ]
    )