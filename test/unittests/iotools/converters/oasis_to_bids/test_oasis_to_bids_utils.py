from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from clinica.iotools.converters.oasis_to_bids.oasis_to_bids_utils import (
    _convert_cdr_to_diagnosis,
    create_sessions_df,
    write_scans_tsv,
    write_sessions_tsv,
)


@pytest.fixture
def clinical_data_path(tmp_path: Path) -> Path:
    clinical_data_path = tmp_path / "clinical"
    _build_clinical_data(clinical_data_path)
    return clinical_data_path


def _build_clinical_data(clinical_data_path: Path) -> None:
    clinical_data_path.mkdir()

    df = pd.DataFrame(
        {
            "ID": ["OAS1_0001_MR1", "OAS1_0002_MR1"],
            "M/F": ["F", "M"],
            "Hand": ["R", "L"],
            "Age": [74, 67],
            "Educ": [2, 2],
            "SES": [3, 3],
            "MMSE": [29, 29],
            "CDR": [0, 0],
            "eTIV": [1344, 1344],
            "nWBV": [0.704, 0.645],
            "ASF": [1.306, 1.100],
            "Delay": [float("nan"), float("nan")],
        }
    )
    df.to_excel(
        clinical_data_path / "oasis_cross-sectional-5708aa0a98d82080.xlsx", index=False
    )


@pytest.fixture
def sessions_path_success(tmp_path: Path) -> Path:
    sessions_path_success = tmp_path / "spec"
    _build_spec_sessions_success(sessions_path_success)
    return sessions_path_success


def _build_spec_sessions_success(sessions_path_success: Path) -> None:
    sessions_path_success.mkdir()
    spec = pd.DataFrame(
        {
            "BIDS CLINICA": ["cdr_global", "MMS", "diagnosis", "foo"],
            "ADNI": [np.nan, np.nan, np.nan, "foo"],
            "OASIS": ["CDR", "MMSE", "CDR", np.nan],
            "OASIS location": [
                "oasis_cross-sectional-5708aa0a98d82080.xlsx",
                "oasis_cross-sectional-5708aa0a98d82080.xlsx",
                "oasis_cross-sectional-5708aa0a98d82080.xlsx",
                np.nan,
            ],
        }
    )
    spec.to_csv(sessions_path_success / "sessions.tsv", index=False, sep="\t")


@pytest.fixture
def sessions_path_error(tmp_path: Path) -> Path:
    sessions_path_error = tmp_path / "spec"
    _build_spec_sessions_error(sessions_path_error)
    return sessions_path_error


def _build_spec_sessions_error(sessions_path_error: Path) -> None:
    sessions_path_error.mkdir()
    spec = pd.DataFrame(
        {
            "BIDS CLINICA": ["foo"],
            "OASIS": ["foo"],
            "OASIS location": [
                "foo.csv",
            ],
        }
    )
    spec.to_csv(sessions_path_error / "sessions.tsv", index=False, sep="\t")


@pytest.fixture
def bids_dir(tmp_path: Path) -> Path:
    bids_dir = tmp_path / "BIDS"
    _build_bids_dir(bids_dir)
    return bids_dir


def _build_bids_dir(bids_dir: Path) -> None:
    (bids_dir / "sub-OASIS10001" / "ses-M000").mkdir(parents=True)
    (bids_dir / "sub-OASIS10001" / "ses-M006").mkdir(parents=True)
    (bids_dir / "sub-OASIS10002" / "ses-M000").mkdir(parents=True)


@pytest.fixture
def expected() -> pd.DataFrame:
    expected = {
        "sub-OASIS10001": {
            "session_id": "ses-M000",
            "cdr_global": 0,
            "MMS": 29,
            "diagnosis": "CN",
        },
        "sub-OASIS10002": {
            "session_id": "ses-M000",
            "cdr_global": 0,
            "MMS": 29,
            "diagnosis": "CN",
        },
    }

    expected = pd.DataFrame.from_dict(expected).T
    expected.index.names = ["BIDS ID"]

    return expected


def test_create_sessions_df_success(
    tmp_path,
    clinical_data_path: Path,
    sessions_path_success: Path,
    expected: pd.DataFrame,
):
    result = create_sessions_df(
        clinical_data_path,
        sessions_path_success,
        ["sub-OASIS10001", "sub-OASIS10002"],
    )
    assert_frame_equal(expected, result, check_like=True, check_dtype=False)


def test_create_sessions_df_missing_clinical_data(
    tmp_path,
    clinical_data_path: Path,
    sessions_path_success: Path,
    expected: pd.DataFrame,
):
    result = create_sessions_df(
        clinical_data_path,
        sessions_path_success,
        ["sub-OASIS10001", "sub-OASIS10002", "sub-OASIS10004"],
    )
    missing_line = pd.DataFrame.from_dict(
        {
            "sub-OASIS10004": {
                "session_id": "ses-M000",
                "diagnosis": "n/a",
                "cdr_global": "n/a",
                "MMS": "n/a",
            }
        }
    ).T
    missing_line.index.names = ["BIDS ID"]

    expected = pd.concat([expected, missing_line])
    assert_frame_equal(expected, result, check_like=True, check_dtype=False)


def test_create_sessions_df_file_not_found(
    tmp_path,
    clinical_data_path: Path,
    sessions_path_error: Path,
):
    with pytest.raises(FileNotFoundError):
        create_sessions_df(
            clinical_data_path,
            sessions_path_error,
            ["sub-OASIS10001", "sub-OASIS10002"],
        )


def test_write_sessions_tsv(
    tmp_path,
    clinical_data_path: Path,
    bids_dir: Path,
    sessions_path_success: Path,
    expected: pd.DataFrame,
):
    sessions = create_sessions_df(
        clinical_data_path,
        sessions_path_success,
        ["sub-OASIS10001", "sub-OASIS10002"],
    )
    write_sessions_tsv(bids_dir, sessions)
    sessions_files = list(bids_dir.rglob("*.tsv"))

    assert len(sessions_files) == 2
    for file in sessions_files:
        assert_frame_equal(
            pd.read_csv(file, sep="\t").reset_index(drop=True),
            expected.loc[[file.parent.name]].reset_index(drop=True),
            check_like=True,
            check_dtype=False,
        )
        assert file.name == f"{file.parent.name}_sessions.tsv"


def test_write_scans_tsv(tmp_path, bids_dir: Path) -> None:
    image_path = (
        bids_dir
        / "sub-OASIS10001"
        / "ses-M000"
        / "anat"
        / "sub-OASIS10001_ses-M000_T1.nii.gz"
    )
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    write_scans_tsv(bids_dir)

    for session_path in bids_dir.rglob("ses-M*"):
        tsv_path = list(session_path.rglob("*scans.tsv"))
        if session_path.name != "ses-M000":
            assert not tsv_path
        else:
            assert len(tsv_path) == 1
            sub = session_path.parent.name
            assert (
                tsv_path[0] == bids_dir / sub / "ses-M000" / f"{sub}_ses-M000_scans.tsv"
            )
            file = pd.read_csv(tsv_path[0], sep="\t")
            if sub == "sub-OASIS10001":
                assert file["filename"].loc[0] == f"anat/{image_path.name}"
            elif sub == "sub-OASIS10002":
                assert file.empty


@pytest.mark.parametrize(
    "cdr,diagnosis",
    [
        (0, "CN"),
        (12, "AD"),
        (-2, "n/a"),
        ("n/a", "n/a"),
        ("foo", "n/a"),
    ],
)
def test_convert_cdr_to_diagnosis(cdr, diagnosis):
    assert diagnosis == _convert_cdr_to_diagnosis(cdr)
