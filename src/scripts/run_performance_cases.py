from dataclasses import dataclass
import glob
import os
import subprocess
from pathlib import Path
from typing import Optional
import datargs
from datargs import parse


# run the testcases in the currently activated virtual environment
@dataclass
class Config:
    test_case_folder: str = datargs.arg(
        default=None, help="Folder containing the test cases", metavar="path"
    )
    output_folder_parent: str = datargs.arg(
        default=None, help="Parent folder for the outputs", metavar="path"
    )
    version: Optional[str] = datargs.arg(
        default=None,
        help="Optionally provide a version. Will default to the git tag if not provided.",
        metavar="str",
    )


def run(config: Config) -> None:
    test_case_folder = Path(config.test_case_folder)
    output_folder_parent = Path(config.output_folder_parent)
    version = config.version
    if version is None:
        version = subprocess.run(
            ["run_rncct", "--version"], capture_output=True, text=True
        ).stdout.strip()
    test_case_ids = os.listdir(test_case_folder)

    # is this is source run or executable run, we can determine the venv path differently. For source run, we can get the venv path from the version of ncct package.

    for case_id in test_case_ids:
        case_path = test_case_folder / case_id
        output_path = output_folder_parent / version / case_id
        if output_path.exists():
            continue

        output_path.mkdir(parents=True, exist_ok=True)

        if len(glob.glob(str(case_path / "*.nii"))) == 1:
            case_path = Path(glob.glob(str(case_path / "*.nii"))[0])
        elif len(glob.glob(str(case_path / "*.nii.gz"))) == 1:
            case_path = Path(glob.glob(str(case_path / "*.nii.gz"))[0])
        # now run the rncct process

        subprocess.run(
            [
                "run_rncct",
                "--input",
                str(case_path),
                "--output",
                str(output_path),
                "--thin2thick",
            ]
        )

        print(f"Completed rNCCT for case {case_id} in version {version}")


def main() -> None:
    config = parse(Config)
    run(config)


if __name__ == "__main__":
    main()
