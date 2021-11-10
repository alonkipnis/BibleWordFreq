# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal

from biblical_scripts.pipelines.OSHB import pipeline as oshb
from biblical_scripts.pipelines.data_engineering import pipeline as de
from biblical_scripts.pipelines.sim import pipeline as ds
from biblical_scripts.pipelines.sim_val import pipeline as ds_val
from biblical_scripts.pipelines.sim_full import pipeline as sim
from biblical_scripts.pipelines.bootstrap import pipeline as bs
from biblical_scripts.pipelines.plotting import pipeline as plot
from biblical_scripts.pipelines.plotting_BS import pipeline as plot_BS
from biblical_scripts.pipelines.reporting import pipeline as report
from biblical_scripts.pipelines.chunk_len import pipeline as chunk_len


class ProjectHooks:
    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """
        de_pipeline = de.create_pipeline()
        sim_only_pipeline = ds.create_pipeline()
        ds_val_pipeline = ds_val.create_pipeline()
        sim_pipeline = sim.create_pipeline()
        oshb_pipeline = oshb.create_pipeline()
        bs_pipeline = bs.create_pipeline()
        plot_pipeline = plot.create_pipeline()
        plot_BS_pipeline = plot_BS.create_pipeline()
        report_pipeline = report.create_pipeline()
        chunk_len_pipeline = chunk_len.create_pipeline()

        return {
            "oshb" : oshb_pipeline,
            "de" : de_pipeline,
            "sim_val" : ds_val_pipeline,
            "sim_only" : sim_only_pipeline,
            "sim_full" : sim_pipeline,
            "plot" : plot_pipeline,
            "sim_bs" : bs_pipeline,
            "plot_bs" : plot_BS_pipeline,
            "report" : report_pipeline,
            "chunk_len" : chunk_len_pipeline,
            "all" : oshb_pipeline+de_pipeline+sim_pipeline+plot_pipeline+report_pipeline,
            "de_sim" : de_pipeline + sim_only_pipeline, 
            "__default__" :  de_pipeline+sim_pipeline+plot_pipeline+report_pipeline}

    @hook_impl
    def register_config_loader(self, conf_paths: Iterable[str]) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )
