from __future__ import annotations
from typing import Optional

import click

from clinica import option
from clinica.pipelines import cli_param
from clinica.pipelines.engine import clinica_pipeline

from .apply_transform_pipeline import UtilsApplyTransform

pipeline_name = "utils-apply-transform"


@clinica_pipeline
@click.command(name=pipeline_name)
@cli_param.argument.bids_directory
@cli_param.argument.caps_directory
@cli_param.option_group.pipeline_specific_options
@cli_param.option.option(
    "--from",
    "from_mode",
    type=click.Choice(["t1-linear", "flair-linear"], case_sensitive=True),
    default="t1-linear",
    show_default=True,
    help="Which pipeline's transform to use.",
)
@cli_param.option.option(
    "--mask-suffix",
    "mask_suffix",
    default="desc-tumor_mask",
    show_default=True,
    help=(
        "Mask filename suffix (e.g., 'desc-tumor_mask' to match "
        "sub-XX_ses-YY_desc-tumor_mask.nii.gz)."
    ),
)
@cli_param.option.option(
    "--space",
    "space",
    default="MNI152NLin2009cSym",
    show_default=True,
    help="Target space to resample into.",
)
@cli_param.option.option(
    "--use-antspy",
    "use_antspy",
    is_flag=True,
    help="(Reserved) Use ANTsPy instead of ANTs.",
)
@cli_param.option.option(
    "-ui",
    "--uncropped_image",
    is_flag=True,
    help="Do not crop the image with template (match t1-linear/flair-linear behavior).",
)
@cli_param.option_group.common_pipelines_options
@cli_param.option.subjects_sessions_tsv
@cli_param.option.working_directory
@option.global_option_group
@option.n_procs
@cli_param.option.caps_name

def cli(
    bids_directory: str,
    caps_directory: str,
    from_mode: str = "t1-linear",
    mask_suffix: str = "desc-tumor_mask",
    space: str = "MNI152NLin2009cSym",
    use_antspy: bool = False,
    uncropped_image: bool = False,
    subjects_sessions_tsv: Optional[str] = None,
    working_directory: Optional[str] = None,
    n_procs: Optional[int] = None,
    caps_name: Optional[str] = None,
) -> None:
    """Apply a precomputed transform to a segmentation mask (nearest-neighbor).

    The transform is retrieved from a previous pipeline (t1-linear / flair-linear / dwi-chain)
    and applied to a BIDS/derivatives mask (e.g., ``desc-tumor_mask.nii.gz``), writing a
    CAPS-compliant output under ``subjects/<sub>/<ses>/labels``. Supports optional cropping
    consistent with t1-linear / flair-linear via ``--uncropped_image``.
    """
    from networkx import Graph
    from clinica.utils.ux import print_end_pipeline

    parameters = {
        "from": from_mode,
        "mask_suffix": mask_suffix,
        "space": space,
        "use_antspy": use_antspy,
        "uncropped_image": uncropped_image,
    }

    pipeline = UtilsApplyTransform(
        bids_directory=bids_directory,
        caps_directory=caps_directory,
        tsv_file=subjects_sessions_tsv,
        parameters=parameters,
        name=pipeline_name,
        use_antspy=use_antspy,
        caps_name=caps_name,
    )

    # Working directory and MultiProc config
    if working_directory:
        pipeline.base_dir = working_directory

    exec_pipeline = (
        pipeline.run(plugin="MultiProc", plugin_args={"n_procs": int(n_procs)})
        if n_procs
        else pipeline.run()
    )

    if isinstance(exec_pipeline, Graph):
        print_end_pipeline(pipeline_name, pipeline.base_dir, pipeline.base_dir_was_specified)


if __name__ == "__main__":
    cli()