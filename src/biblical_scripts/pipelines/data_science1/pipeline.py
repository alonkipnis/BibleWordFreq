#pipeline: data science 1 (select features)

from kedro.pipeline import node, Pipeline
from biblical_scripts.pipelines.data_science1.nodes import (build_reduced_vocab, build_model, model_predict, evaluate_accuracy, filter_by_author)
from biblical_scripts.pipelines.data_engineering.nodes import add_convert
from biblical_scripts.pipelines.plotting.nodes import plot_sim

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:known_authors"],
             outputs="data",
             name="filter_by_author"
            ),
        node(func=build_reduced_vocab, 
             inputs=["data", "vocabulary", "params:reduction_method", "params:model"],
             outputs="reduced_vocabulary1",
             name="feature_selection"
            ),
        node(func=add_convert,
             inputs=["reduced_vocabulary1", "oshb_parsed"],
             outputs="reduced_vocabulary",
             name="vocab_conversion"
            ),
        node(func=build_model,
             inputs=["data","reduced_vocabulary", "params:model"],
             outputs="model",
             name="build_model"
            ),
        node(func=model_predict, 
             inputs=["data_proc", "model"],
             outputs="sim_res",
             name="model_predict",
            ),
        node(func=evaluate_accuracy, 
             inputs=["sim_res", "params:report", "parameters"],
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