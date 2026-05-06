# hypodensity

Non-contrast CT (NCCT) hypodensity quantification relative to the contralateral mirror region.

Given a head NCCT scan, `hypodensity` registers the brain to a standard template, computes a flip-to-self registration, and produces a map of Hounsfield Unit (HU) depression relative to the contralateral hemisphere. The pipeline is designed for stroke-related analysis and outputs lesion volume estimates at configurable HU thresholds.

## Pipeline overview

DICOM/NIfTI input → template registration → brain masking → flip-to-self registration → CSF-aware masking → Gaussian smoothing → HU depression map → lesion volume summaries

## Installation

Requires Python 3.10 or later.

```bash
git clone git@github.com:hypodensity/hypodensity.git
cd hypodensity
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
run_rncct -i /path/to/dicom_folder -o /path/to/output_folder
```

NIfTI input is also supported:

```bash
run_rncct -i /path/to/scan.nii.gz -o /path/to/output_folder
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | — | Input: DICOM folder (single series) or NIfTI file |
| `-o` | — | Output folder (DICOM format) |
| `-t` | off | Resample thin slices (<3 mm) to ~5 mm spacing |
| `--thresholds` | `1,4.9` | Comma-separated HU depression thresholds for volume calculation |
| `--colormap_range` | `3,10` | Colormap percentile range for overlay rendering |
| `--xy_std` | `5.0` | Gaussian smoothing sigma in the XY plane (in mm) |
| `--z_std` | `5.0` | Gaussian smoothing sigma along Z (in mm) |
| `--max_accept_HU` | `45.0` | Ignore voxels above this HU value |
| `-d` | off | Debug mode: produce additional outputs (e.g. registration movies) |
| `--version` | — | Print version and exit |

## Output

The output folder contains a DICOM series with the HU depression overlay alongside a summary of lesion volumes at each configured threshold.

## Citation

If you use this software in research, please cite:

```bibtex
@article{christensen2023hypodensity,
  title   = {Semiautomated Detection of Early Infarct Signs on Noncontrast CT Improves Interrater Agreement},
  author  = {Christensen, Soren and Demeestere, Jelle and Verhaaren, Benjamin F J and Heit, Jeremy J and Von Stein, Erica Leah and Madill, Evan S and Loube, Deanne Kennedy and Dugue, Rachelle and Rengarajan, Sophie and Mlynash, Michael and Albers, Gregory W and Lemmens, Robin and Lansberg, Maarten G},
  journal = {Stroke},
  volume  = {54},
  number  = {12},
  pages   = {3090--3096},
  year    = {2023},
  doi     = {10.1161/STROKEAHA.123.044058}
}
```

## Help
Please use the dscussion tab in the github link


## License

MIT — see [LICENSE](LICENSE).
