import sys
from importlib.metadata import PackageNotFoundError, version

from datargs import parse

from ncct_utils.rncct_config import rNCCTConfig
from ncct_utils.rncct_process import run_config


def get_package_version() -> str:
    """Get the package version from metadata."""
    try:
        return version("hypodensity")
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    rncct_config = parse(rNCCTConfig)

    # Handle --version flag
    if rncct_config.version:
        print(f"{get_package_version()}")
        sys.exit(0)

    run_config(rncct_config)


if __name__ == "__main__":
    main()
