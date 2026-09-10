from __future__ import annotations
from typing import List, Optional, Tuple

from pathlib import Path
import json
from nipype import config

from clinica.pipelines.engine import Pipeline

# Keep nipype folders short
cfg = dict(execution={"parameterize_dirs": False})
config.update_config(cfg)


def _caps_flair_linear_paths(
    caps_dir: Path, sub: str, ses: str, space: str
) -> Tuple[Path, Path]:
    base = caps_dir / "subjects" / sub / ses / "flair_linear"
    affine = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_affine.mat"))
    ref = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_FLAIR.nii.gz"))
    if not affine or not ref:
        raise FileNotFoundError(
            f"flair-linear outputs not found for {sub} {ses} in space {space}."
        )
    return affine[0], ref[0]


def _caps_t1_linear_paths(
    caps_dir: Path, sub: str, ses: str, space: str
) -> Tuple[Path, Path]:
    base = caps_dir / "subjects" / sub / ses / "t1_linear"
    affine = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_affine.mat"))
    ref = list(base.glob(f"{sub}_{ses}_space-{space}_res-1x1x1_T1w.nii.gz"))
    if not affine or not ref:
        raise FileNotFoundError(
            f"t1-linear outputs not found for {sub} {ses} in space {space}."
        )
    return affine[0], ref[0]


def _find_input_image(bids_dir: Path, sub: str, ses: str, input_suffix: str) -> Path:
    """
    Find the BIDS image to be transformed.

    Example patterns:
      BIDS/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz
      BIDS/sub-01/ses-01/anat/sub-01_ses-01_desc-something_T1w.nii.gz

    `input_suffix` is the trailing part after 'sub-XX_ses-YY_', e.g. 'T1w' or
    'desc-something_T1w'.
    """
    anat = bids_dir / sub / ses / "anat"
    cand = list(anat.glob(f"{sub}_{ses}_{input_suffix}.nii.gz"))
    if cand:
        return cand[0]

    # fallback: search in derivatives (if the image is produced by some other pipeline)
    deriv = bids_dir / "derivatives"
    cand = list(deriv.rglob(f"{sub}_{ses}_{input_suffix}.nii.gz"))
    if cand:
        return cand[0]

    raise FileNotFoundError(
        f"Input image not found for {sub} {ses}: suffix '{input_suffix}'. "
        "Looked in anat/ and derivatives/."
    )


class UtilsApplyTransform(Pipeline):
    """Reapply an existing affine transform (from t1-linear / flair-linear) to a BIDS image.

    - Interpolation: Linear (trilinear)
    - Input: BIDS image specified by `input_suffix`
    - Output folder: CAPS/subjects/<sub>/<ses>/custom/
    - Output name: <sub>_<ses>_space-<space>_<output-suffix>.nii.gz
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
        super().__init__(
            bids_directory, caps_directory, tsv_file, name=name, parameters=parameters
        )
        # Defaults
        self.parameters.setdefault("from", "t1-linear")
        self.parameters.setdefault("input_suffix", "T1w")
        self.parameters.setdefault("output_suffix", "desc-noBias")
        self.parameters.setdefault("space", "MNI152NLin2009cSym")
        self.parameters.setdefault("uncropped_image", False)
        self.parameters.setdefault("output_subfolder", "custom")
        self.parameters.setdefault("interpolation", "Linear")

        self.use_antspy = use_antspy
        self.caps_name = caps_name

    def _check_custom_dependencies(self) -> None:
        """Check dependencies that can't be listed in info.json."""
        # antsApplyTransforms (ANTs) via Nipype must be on PATH; Nipype will raise otherwise.
        return

    def get_input_fields(self) -> List[str]:
        return ["in_file", "reference", "transform", "subject", "session"]

    def get_output_fields(self) -> List[str]:
        return ["out_file"]

    def _build_input_node(self) -> None:
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        from clinica.utils.stream import cprint
        from clinica.utils.ux import print_images_to_process

        bids = Path(self.bids_directory)
        caps = Path(self.caps_directory)
        mode_from: str = self.parameters["from"]
        input_suffix: str = self.parameters["input_suffix"]
        space: str = self.parameters["space"]
        modality = input_suffix.split("_")[-1]

        # Collect parallel lists of inputs
        in_files: list[str] = []
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

            in_img = _find_input_image(bids, sub, ses, input_suffix)
            in_files.append(str(in_img))
            refs.append(str(ref))
            xforms.append(str(affine))
            subs.append(sub)
            sess.append(ses)

        if len(subs):
            print_images_to_process(subs, sess)
            cprint("Applying affine to image with Linear {self.parameters['interpolation']} interpolation…")

        read_input_node = npe.Node(
            name="LoadingCLIArguments",
            iterables=[
                ("in_file", in_files),
                ("reference", refs),
                ("transform", xforms),
                ("subject", subs),
                ("session", sess),
            ],
            synchronize=True,
            interface=nutil.IdentityInterface(fields=self.get_input_fields()),
        )

        self.connect(
            [
                (
                    read_input_node,
                    self.input_node,
                    [
                        ("in_file", "in_file"),
                        ("reference", "reference"),
                        ("transform", "transform"),
                        ("subject", "subject"),
                        ("session", "session"),
                    ],
                )
            ]
        )

    def _build_core_nodes(self) -> None:
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        from nipype.interfaces.ants import ApplyTransforms
        from clinica.pipelines.tasks import crop_nifti_using_t1_mni_template_task

        space = self.parameters["space"]
        output_suffix = self.parameters["output_suffix"]
        uncropped = bool(self.parameters.get("uncropped_image", False))
        output_subfolder = self.parameters.get("output_subfolder", "custom")
        interpolation = self.parameters.get(
            "interpolation",
            "Linear",
        )
        # 1) Build paths: final CAPS path + a temp file in working dir
        def build_paths(
            caps_dir,
            base_dir,
            subject,
            session,
            space,
            output_suffix,
            modality_suffix,   # e.g. "FLAIR" or "T1w"
            source_image,
            reference,
            transform,
            from_mode,
            interpolation,
            output_subfolder,
        ):

            from pathlib import Path
            import json

            caps_dir = Path(caps_dir)
            base_dir = Path(base_dir) if base_dir else caps_dir

            out_dir = caps_dir / "subjects" / subject / session / output_subfolder
            out_dir.mkdir(parents=True, exist_ok=True)

            # ---- UNCROPPED FILENAME ----
            uncropped_name = (
                f"{subject}_{session}_space-{space}_{output_suffix}_{modality_suffix}.nii.gz"
            )
            uncropped_final = out_dir / uncropped_name

            # ---- CROPPED FILENAME ----
            cropped_name = (
                f"{subject}_{session}_space-{space}_desc-Crop_res-1x1x1_{output_suffix}_{modality_suffix}.nii.gz"
            )
            cropped_final = out_dir / cropped_name

            # Temporary directory
            tmp_dir = Path(base_dir) / "tmp_utils_apply_transform"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_out = tmp_dir / f"{subject}_{session}_tmp_apply.nii.gz"

            # Metadata JSON (same for cropped & uncropped)
            meta = {
                "Sources": [str(source_image)],
                "Transform": str(transform),
                "Reference": str(reference),
                "Interpolation": interpolation,
                "Space": space,
                "Pipeline": f"utils-apply-transform (from {from_mode})",
            }

            # JSON written next to the uncropped file (Clinica convention)
            json_path = uncropped_final.with_suffix("").with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)

            return (
                str(tmp_out),
                str(uncropped_final),
                str(cropped_final),
                str(out_dir),
                str(tmp_dir),
            )


        mk = npe.Node(
            interface=nutil.Function(
                input_names=[
                    "caps_dir", "base_dir", "subject", "session", "space",
                    "output_suffix","modality_suffix", "source_image", "reference", "transform",
                    "from_mode","interpolation", "output_subfolder",
                ],
                output_names=[
                    "tmp_path",
                    "uncropped_final",
                    "cropped_final",
                    "final_dir",
                    "tmp_dir",
                ],
                function=build_paths,
            ),
            name="buildPaths",
        )
        mk.inputs.caps_dir = str(self.caps_directory)
        mk.inputs.base_dir = str(self.base_dir) if self.base_dir else None
        mk.inputs.space = space
        mk.inputs.output_suffix = output_suffix
        mk.inputs.from_mode = self.parameters["from"]
        mk.inputs.output_subfolder = output_subfolder
        mk.inputs.modality_suffix = self.parameters["input_suffix"].split("_")[-1]
        mk.inputs.interpolation = interpolation
        # 2) antsApplyTransforms -> temp file (Linear interpolation)
        apply = npe.Node(name="antsApplyTransforms", interface=ApplyTransforms())
        apply.inputs.dimension = 3
        apply.inputs.interpolation = interpolation  # trilinear interpolation
        apply.inputs.default_value = 0

        # 3) Optional crop (like t1-linear/flair-linear)
        crop = npe.Node(
            name="cropImage",
            interface=nutil.Function(
                function=crop_nifti_using_t1_mni_template_task,
                input_names=["input_image", "output_path"],
                output_names=["output_image"],
            ),
        )

        # 4) Select/copy final: pick cropped or tmp depending on flag; copy to final CAPS path
        def select_and_copy(tmp_path, cropped_path, uncropped_final, cropped_final, uncropped_image):
            import shutil
            import os

            if uncropped_image:
                # UNCROPPED MODE
                if not os.path.isfile(tmp_path):
                    raise FileNotFoundError(f"Missing uncropped intermediate file: {tmp_path}")
                shutil.copyfile(tmp_path, uncropped_final)
                return uncropped_final

            else:
                # CROPPED MODE
                if not os.path.isfile(cropped_path):
                    raise FileNotFoundError(f"Missing cropped image: {cropped_path}")
                shutil.copyfile(cropped_path, cropped_final)
                return cropped_final



        finalize = npe.Node(
            interface=nutil.Function(
                input_names=[
                    "tmp_path",
                    "cropped_path",
                    "uncropped_final",
                    "cropped_final",
                    "uncropped_image",
                ],
                output_names=["out_file"],
                function=select_and_copy,
            ),
            name="finalizeOutput",
        )
        finalize.inputs.uncropped_image = uncropped

        # Wiring
        self.connect([
            # 1. Build paths
            (self.input_node, mk, [
                ("subject",  "subject"),
                ("session",  "session"),
                ("in_file",  "source_image"),
                ("reference", "reference"),
                ("transform", "transform"),
            ]),

            # 2. Apply transform → tmp_path
            (self.input_node, apply, [
                ("in_file", "input_image"),
                ("reference", "reference_image"),
                ("transform", "transforms"),
            ]),
            (mk, apply, [("tmp_path", "output_image")]),

            # 3. Crop: output_path = crop_dir
            (mk, crop, [("tmp_dir", "output_path")]),
            (apply, crop, [("output_image", "input_image")]),

            # 4. Finalize copy
            (apply, finalize, [("output_image", "tmp_path")]),
            (crop, finalize, [("output_image", "cropped_path")]),

            # 5. Provide final filenames
            (mk, finalize, [
                ("uncropped_final", "uncropped_final"),
                ("cropped_final",   "cropped_final"),
            ]),
        ])


    def _build_output_node(self) -> None:
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe

        # Connect the finalizeOutput node to the pre-created output_node.
        out_node = npe.Node(
            nutil.IdentityInterface(fields=self.get_output_fields()),
            name="OutputCollector",
        )

        self.connect([
            (self.get_node("finalizeOutput"), out_node, [("out_file", "out_file")]),
        ])

        self.connect([(out_node, self.output_node, [("out_file", "out_file")])])
