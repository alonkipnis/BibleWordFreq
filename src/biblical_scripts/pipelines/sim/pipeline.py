#pipeline: data science (sim only)

from kedro.pipeline import node, Pipeline

from biblical_scripts.pipelines.plotting.nodes import plot_sim
from biblical_scripts.pipelines.data_engineering.nodes import add_convert
from biblical_scripts.pipelines.reporting.nodes import  (
     report_table_known, report_table_unknown)

from .nodes import (build_model, model_predict, filter_by_author)

def create_pipeline(**kwargs):
    return Pipeline(
        [
        node(func=filter_by_author, 
             inputs=["data_proc", "params:known_authors", "params:unk_authors"],
             outputs="data_filtered",
             name="filter_by_author"
            ),
        node(func=build_model,
             inputs=["data_filtered", "vocabulary", "params:model"],
             outputs=["model", "reduced_vocabulary0"],
             name="build_model"
            ),
        node(func=add_convert,
             inputs=["reduced_vocabulary0", "oshb_parsed"],
             outputs="reduced_vocabulary",
             name="translate_vocab"
            ),
        node(func=model_predict, 
             inputs=["data_proc", "model"],
             outputs="sim_res",
             name="model_predict",
            ),
        node(func=report_table_known,
             inputs=["sim_res", "params:report"],
             outputs="sim_table_report_known",
             name="report_table_known",
            ),
        node(func=report_table_unknown,
             inputs=["sim_res", "params:report"],
             outputs="sim_table_report_unknown",
             name="report_table_unknown",
            ),
        node(func=plot_sim,
             inputs=["sim_res", "params:report"],
             outputs=None,
             name="plot_sim"
            ),
        ], tags='basic similarity'
    )
