import shutil
import tempfile

import pytest
from pathlib import Path
import subprocess
from ncct_utils.rncct_config import rNCCTConfig
from ncct_utils.rncct_process import run_config

INPUT = str(Path(__file__).parent / "data" / "SIM/SIM.tar.zstd")
OUTPUT = str(Path(__file__).parent.parent / "OP")


@pytest.fixture(scope="session")
def pipeline_output() -> Path:
    tdir = tempfile.mkdtemp(prefix="ncct_pipeline_test_")
    subprocess.run(["tar", "-I", "zstd", "-xf", INPUT, "-C", tdir], check=True)

    cfg = rNCCTConfig(
        input=os.path.join(tdir, "SIM"),
        output=OUTPUT,
        caching=True,
        debug=True,
    )
    run_config(cfg)
    shutil.rmtree(tdir)
    return Path(OUTPUT)
