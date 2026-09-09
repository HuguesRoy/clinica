from __future__ import annotations
from typing import List, Optional, Tuple

from pathlib import Path
import json
from nipype import config

from clinica.pipelines.engine import Pipeline

# Keep nipype folders short
cfg = dict(execution={"parameterize_dirs": False})
config.update_config(cfg)


def _caps_flair_linear_paths(caps_dir: Path, sub: str, ses: str, space: str) -> Tuple[Path, Path]:
    base = caps_dir / "subjects" / sub / ses / "flair_linear"
    affine = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_affine.mat"))
    ref = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_FLAIR.nii.gz"))
    if not affine or not ref:
        raise FileNotFoundError(
            f"flair-linear outputs not found for {sub} {ses} in space {space}."
        )
    return affine[0], ref[0]


def _caps_t1_linear_paths(caps_dir: Path, sub: str, ses: str, space: str) -> Tuple[Path, Path]:
    base = caps_dir / "subjects" / sub / ses / "t1_linear"
    affine = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_affine.mat"))
    ref = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_T1w.nii.gz"))
    if not affine or not ref:
        raise FileNotFoundError(
            f"t1-linear outputs not found for {sub} {ses} in space {space}."
        )
    return affine[0], ref[0]

def _find_mask(bids_dir: Path, sub: str, ses: str, mask_suffix: str) -> Path:
    # e.g. BIDS/sub-01/ses-01/anat/sub-01_ses-01_desc-tumor_mask.nii.gz
    anat = bids_dir / sub / ses / "anat"
    cand = list(anat.glob(f"{sub}_{ses}_{mask_suffix}.nii.gz"))
    if cand:
        return cand[0]
    # try derivatives fallback
    deriv = bids_dir / "derivatives"
    cand = list(deriv.rglob(f"{sub}_{ses}_{mask_suffix}.nii.gz"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"Mask not found for {sub} {ses}: suffix '{mask_suffix}'.")



class UtilsApplyTransform(Pipeline):
    """Apply an existing affine transform (from t1-linear / flair-linear / dwi-chain) to a segmentation mask.

    - Interpolation: NearestNeighbor (label-preserving)
    - Output location: CAPS/subjects/<sub>/<ses>/labels/
    - Output name: <sub>_<ses>_space-<space>_<mask-suffix>.nii.gz
    """

    def __init__(
        self,
        bids_directory: str | Path,
        caps_directory: str | Path,
        tsv_file: Optional[str] = None,
        parameters: Optional[dict] = None,
        name: str = "utils-apply-transform",
        use_antspy: bool = False,
        caps_name: Optional[str] = None,
    ) -> None:
        parameters = parameters or {}
        super().__init__(bids_directory, caps_directory, tsv_file, name=name, parameters=parameters)
        # defaults
        self.parameters.setdefault("from", "t1-linear")
        self.parameters.setdefault("mask_suffix", "desc-tumor_mask")
        self.parameters.setdefault("space", "MNI152NLin2009cSym")
        self.use_antspy = use_antspy
        self.caps_name = caps_name

    def _check_custom_dependencies(self) -> None:
        """Check dependencies that can't be listed in info.json."""
        # antsApplyTransforms (ANTs) via Nipype must be on PATH; Nipype will raise otherwise.
        return

    def get_input_fields(self) -> List[str]:
        return ["mask", "reference", "transform", "subject", "session"]

    def get_output_fields(self) -> List[str]:
        return ["out_mask"]

    def _build_input_node(self) -> None:
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        from clinica.utils.stream import cprint
        from clinica.utils.ux import print_images_to_process

        bids = Path(self.bids_directory)
        caps = Path(self.caps_directory)
        mode_from: str = self.parameters["from"]
        mask_suffix: str = self.parameters["mask_suffix"]
        space: str = self.parameters["space"]

        # Collect parallel lists of inputs like in PETLinear
        masks: list[str] = []
        refs: list[str] = []
        xforms: list[str] = []
        subs: list[str] = []
        sess: list[str] = []
        for sub, ses in zip(self.subjects, self.sessions):
            if mode_from == "t1-linear":
                affine, ref = _caps_t1_linear_paths(caps, sub, ses, space)
            elif mode_from == "flair-linear":
                affine, ref = _caps_flair_linear_paths(caps, sub, ses, space)
            else:
                raise ValueError(f"Unsupported --from mode: {mode_from}")
            mask = _find_mask(bids, sub, ses, mask_suffix)
            masks.append(str(mask))
            refs.append(str(ref))
            xforms.append(str(affine))
            subs.append(sub)
            sess.append(ses)

        if len(subs):
            print_images_to_process(subs, sess)
            cprint("Applying affine to mask with NearestNeighbor interpolation…")

        read_input_node = npe.Node(
            name="LoadingCLIArguments",
            iterables=[
                ("mask", masks),
                ("reference", refs),
                ("transform", xforms),
                ("subject", subs),
                ("session", sess),
            ],
            synchronize=True,
            interface=nutil.IdentityInterface(fields=self.get_input_fields()),
        )
        self.connect([(read_input_node, self.input_node, [("mask", "mask"),
                                                          ("reference", "reference"),
                                                          ("transform", "transform"),
                                                          ("subject", "subject"),
                                                          ("session", "session")])])

    def _build_core_nodes(self) -> None:
        import os
        import shutil
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        from nipype.interfaces.ants import ApplyTransforms
        from clinica.pipelines.tasks import crop_nifti_using_t1_mni_template_task

        space = self.parameters["space"]
        mask_suffix = self.parameters["mask_suffix"]
        uncropped = bool(self.parameters.get("uncropped_image", False))

        # 1) Build paths: final CAPS path + a temp file in working dir
        def build_paths(caps_dir, base_dir, subject, session, space,
                mask_suffix, source_mask, reference, transform,
                from_mode):
            from pathlib import Path
            import json

            caps_dir = Path(caps_dir)
            base_dir = Path(base_dir) if base_dir else caps_dir
            out_dir = caps_dir / "subjects" / subject / session / "labels"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_final = out_dir / f"{subject}_{session}_space-{space}_{mask_suffix}.nii.gz"

            tmp_dir = Path(base_dir) / "tmp_utils_apply_transform"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_out = tmp_dir / f"{subject}_{session}_tmp_apply.nii.gz"

            meta = {
                "Sources": [str(source_mask)],
                "Transform": str(transform),
                "Reference": str(reference),
                "Interpolation": "NearestNeighbor",
                "Space": space,
                "Pipeline": f"utils-apply-transform (from {from_mode})",
            }
            with open(out_final.with_suffix("").with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)

            return str(tmp_out), str(out_final), str(out_dir), str(tmp_dir)

        mk = npe.Node(
            interface=nutil.Function(
                input_names=["caps_dir", "base_dir", "subject", "session",
                            "space", "mask_suffix", "source_mask",
                            "reference", "transform", "from_mode"],
                output_names=["tmp_path", "final_path", "final_dir", "tmp_dir"],
                function=build_paths,
            ),
            name="buildPaths",
        )
        mk.inputs.caps_dir = str(self.caps_directory)
        mk.inputs.base_dir = str(self.base_dir) if self.base_dir else None
        mk.inputs.space = space
        mk.inputs.mask_suffix = mask_suffix
        mk.inputs.from_mode = self.parameters["from"]

        # 2) antsApplyTransforms -> temp file
        apply = npe.Node(name="antsApplyTransforms", interface=ApplyTransforms())
        apply.inputs.dimension = 3
        apply.inputs.interpolation = "NearestNeighbor"
        apply.inputs.default_value = 0

        # 3) Optional crop (like t1-linear/flair-linear)
        crop = npe.Node(
            name="cropMask",
            interface=nutil.Function(
                function=crop_nifti_using_t1_mni_template_task,
                input_names=["input_image", "output_path"],
                output_names=["output_image"],
            ),
        )

        # 4) Select/copy final: pick cropped or tmp depending on flag; copy to final CAPS path
        def select_and_copy(tmp_path, cropped_path, final_path, uncropped_image: bool):
            import os
            import shutil

            src = tmp_path if uncropped_image else cropped_path
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Expected file not found: {src}")
            shutil.copyfile(src, final_path)
            return final_path

        finalize = npe.Node(
            interface=nutil.Function(
                input_names=["tmp_path", "cropped_path", "final_path", "uncropped_image"],
                output_names=["out_mask"],
                function=select_and_copy,
            ),
            name="finalizeOutput",
        )
        finalize.inputs.uncropped_image = uncropped

        # Wiring
        self.connect([
            (self.input_node, mk, [("subject", "subject"), ("session", "session"), ("mask", "source_mask"), ("reference", "reference"), ("transform", "transform")]),
            (self.input_node, apply, [("mask", "input_image"), ("reference", "reference_image"), ("transform", "transforms")]),
            (mk, apply, [("tmp_path", "output_image")]),
            # crop tmp into final dir
            (mk, crop, [( "tmp_dir", "output_path" )]),
            (apply, crop, [("output_image", "input_image")]),
            # finalize copy
            (apply, finalize, [("output_image", "tmp_path")]),
            (crop, finalize, [("output_image", "cropped_path")]),
            (mk, finalize, [("final_path", "final_path")]),
        ])

    def _build_output_node(self) -> None:
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe

        # `self.output_node` is created by the base Pipeline, no need to recreate.
        # Connect the finalizeOutput node (created in _build_core_nodes) to it.
        self.connect([
            (self.get_node("finalizeOutput"), self.output_node, [("out_mask", "out_mask")]),
        ])