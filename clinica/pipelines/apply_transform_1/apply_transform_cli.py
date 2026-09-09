from __future__ import annotations
from typing import Optional

import click

from clinica import option
from clinica.pipelines import cli_param
from clinica.pipelines.engine import clinica_pipeline

from .apply_transform_pipeline import UtilsApplyTransform

pipeline_name = "utils-apply-transform-1"


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
    help="Which pipeline's transform to use (t1-linear or flair-linear).",
)
@cli_param.option.option(
    "--input-suffix",
    "input_suffix",
    default="T1w",
    show_default=True,
    help=(
        "BIDS filename suffix of the image to transform, e.g. 'T1w', 'FLAIR', "
        "or 'desc-something_T1w'. "
        "This is the part after 'sub-XX_ses-YY_'."
    ),
)
@cli_param.option.option(
    "--output-suffix",
    "output_suffix",
    default="desc-noBias",
    show_default=True,
    help=(
        "Suffix to embed in the output filename to indicate that this image "
        "is without bias-field correction, e.g. 'desc-noBias'."
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
    "--uncropped-image",
    is_flag=True,
    help=(
        "If set, keep the full field of view (no cropping). "
        "By default, the output is cropped like t1-linear/flair-linear."
    ),
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
    input_suffix: str = "T1w",
    output_suffix: str = "desc-noBias",
    space: str = "MNI152NLin2009cSym",
    use_antspy: bool = False,
    uncropped_image: bool = False,
    subjects_sessions_tsv: Optional[str] = None,
    working_directory: Optional[str] = None,
    n_procs: Optional[int] = None,
    caps_name: Optional[str] = None,
) -> None:
    """Reapply a precomputed linear transform to a BIDS image (T1w / FLAIR / etc.).

    The transform is retrieved from a previous pipeline (t1-linear / flair-linear)
    and applied to a BIDS image specified via ``--input-suffix``. The result is
    written as a CAPS-compliant image under:

        subjects/<sub>/<ses>/custom/

    with a filename of the form:

        <sub>_<ses>_space-<space>_<output-suffix>.nii.gz

    Interpolation is trilinear (ANTs 'Linear'), and an optional cropping step,
    consistent with t1-linear / flair-linear, is applied unless
    ``--uncropped-image`` is given.
    """
    from networkx import Graph
    from clinica.utils.ux import print_end_pipeline

    parameters = {
        "from": from_mode,
        "input_suffix": input_suffix,
        "output_suffix": output_suffix,
        "space": space,
        "use_antspy": use_antspy,
        "uncropped_image": uncropped_image,
        "output_subfolder": "custom",  # per your choice
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
        print_end_pipeline(
            pipeline_name, pipeline.base_dir, pipeline.base_dir_was_specified
        )


if __name__ == "__main__":
    cli()
