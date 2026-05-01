import json
import os
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

import datargs
from datargs import parse
from dataclasses import dataclass


@dataclass
class Config:
    output_folder: str = datargs.arg(
        default=None, help="Parent folder for the outputs", metavar="path"
    )
    folder_a: str = datargs.arg(
        default=None, help="Name of the first folder to compare", metavar="str"
    )
    folder_b: str = datargs.arg(
        default=None, help="Name of the second folder to compare", metavar="str"
    )


def run(config: Config) -> None:

    version0_name = os.path.basename(config.folder_a)
    version1_name = os.path.basename(config.folder_b)
    runs = {
        version0_name: config.folder_a,
        version1_name: config.folder_b,
    }

    output_image_path = os.path.join(
        config.output_folder,
        f"lesion_volume_comparison_{version0_name}_vs_{version1_name}.png",
    )
    # iterate cases
    # collect volumes

    results: dict[str, dict[str, dict[str, Any]]] = {}

    for run_name, run_folder in runs.items():
        case_ids = os.listdir(run_folder)
        # summary_lines: list[str] = []
        for case_id in case_ids:
            case_path = Path(run_folder) / case_id
            summary_file = case_path / "processing_summary.json"
            if not summary_file.exists():
                print(
                    f"Warning, missing summary file for case {case_id} in run {run_name}"
                )
                continue
            with open(summary_file, "r") as f:
                summary_data = json.load(f)
            lesion_volumes = summary_data.get("lesion_volumes", {})

            if run_name not in results:
                results[run_name] = {}
            results[run_name][case_id] = lesion_volumes

    version0_lesion_volumes = {
        cid: vols["4.90"] for cid, vols in results[version0_name].items()
    }
    version1_lesion_volumes = {
        cid: vols["4.90"] for cid, vols in results[version1_name].items()
    }

    shared_ids = set(version0_lesion_volumes.keys()).intersection(
        set(version1_lesion_volumes.keys())
    )
    print(f"Number of shared cases: {len(shared_ids)}")

    plt.plot(
        [version0_lesion_volumes[cid] for cid in shared_ids],
        [version1_lesion_volumes[cid] for cid in shared_ids],
        "o",
    )
    # add id labels to points
    for cid in shared_ids:
        plt.text(
            version0_lesion_volumes[cid],
            version1_lesion_volumes[cid],
            cid,
            fontsize=8,
            alpha=0.7,
        )
    xlim = plt.xlim()
    ylim = plt.ylim()
    minlim = min(xlim[0], ylim[0])
    maxlim = max(xlim[1], ylim[1])
    plt.plot([minlim, maxlim], [minlim, maxlim], "r--")
    plt.xlim(minlim, maxlim)
    plt.ylim(minlim, maxlim)
    plt.xlabel("Lesion volume version 0 (ml) at 4.9%")
    plt.ylabel("Lesion volume version 1 (ml) at 4.9%")
    plt.title("Comparison of lesion volumes between rNCCT versions")
    plt.grid()
    plt.savefig(
        output_image_path,
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    config = parse(Config)
    run(config)
