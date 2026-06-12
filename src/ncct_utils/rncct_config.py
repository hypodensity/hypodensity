from dataclasses import dataclass
from datargs import arg


@dataclass
class rNCCTConfig:
    output: str = arg("-o", default=None, help="output, DICOM format", metavar="url")
    input: str = arg(
        "-i",
        default=None,
        help="either a) input DICOM folder with a single series or b) a nifti file",
        metavar="url",
    )
    caching: bool = arg(
        "-c", default=False, help="Caching mode - this is a dev-option only"
    )
    thin2thick: bool = arg(
        "-t",
        default=False,
        help="Convert thinner than 3 to around 5mm spacing (nearest integer multiple)",
    )
    debug: bool = arg(
        "-d",
        default=False,
        help="Produce additional output, such as registration movies",
    )
    xy_std: float = arg(default=5.0, help="xy gauss sigma (in mm)", metavar="float")
    colormap_range: str = arg(
        default="3,10", help="Colormap range. 3,10 means 3%%-10%%", metavar="float"
    )
    z_std: float = arg(default=5.0, help="z gauss sigma (in mm)", metavar="float")
    max_accept_HU: float = arg(
        default=45.0, help="ignore HU above this value", metavar="float"
    )
    thresholds: str = arg(
        default="1,4.9", help="Thresholds for volume calculation", metavar="float"
    )
    version: bool = arg("--version", default=False, help="Print version and exit")

    def get_thresholds(self) -> list[float]:
        """
        Returns the thresholds as a list of floats
        """
        return [float(t) for t in self.thresholds.split(",")]

    def get_colormap_range(self) -> tuple[float, float]:
        """
        Returns the thresholds as a list of floats
        """
        tmp_list = self.colormap_range.split(",")
        assert len(tmp_list) == 2
        return float(tmp_list[0]), float(tmp_list[1])
