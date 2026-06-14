from __future__ import annotations

import datetime
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.uid import UID as pdUID


class DicomSeries:
    def __init__(
        self,
        series_instance_uid: str,
        slicetol: float = 0.01,
        timedefinition: str = "AcquisitionTime",
        is_enhanced: bool = False,
    ) -> None:
        self.series_instance_uid: str = series_instance_uid
        self.timedefinition: str = timedefinition
        self.seriesdatetime: str | datetime.datetime | None = ""
        self.patientid: str = ""
        self.datasets: List[Tuple[Dataset, str]] = []
        self.sortmatrix: np.ndarray = np.zeros((1, 1))
        self.stepsizes: int | np.ndarray | str = -1
        self.equidistant: int = -1
        self.isophasic: bool = False
        self.slicetol: float = slicetol
        self.nslices: int = -1
        self.nframes: int = -1
        self.seriesdescription: str = ""
        self.modality: str = ""
        self.issorted: bool = False
        self.z_pos: int | np.ndarray = 1
        self.seriesdate_fallback_tostudy: bool = True
        self.is_enhanced: bool = is_enhanced
        self.sortorder: np.ndarray
        self.timingmatrix: np.ndarray = np.zeros((1, 1), int)

    def add_ds(self, ds: Tuple[Dataset, str]) -> None:
        self.datasets.append(ds)
        self.issorted = False

    def calc_proj_z(self, iop: List[float] | List, ipp: List[float] | List) -> int:
        if iop == [] or ipp == []:
            return 0
        iop_list = iop
        ipp_list = ipp

        pos = np.array([float(ipp_list[0]), float(ipp_list[1]), float(ipp_list[2])])
        rowcos = np.array([float(iop_list[0]), float(iop_list[1]), float(iop_list[2])])
        colcos = np.array([float(iop_list[3]), float(iop_list[4]), float(iop_list[5])])

        cp = np.cross(rowcos, colcos)
        return int(np.round(np.dot(cp, pos), 2) * 1000)

    def get_slice_time(self, ds: Dataset) -> datetime.datetime:

        date = datetime.datetime.strptime("19000101", "%Y%m%d")
        time = datetime.datetime.strptime("000000", "%H%M%S")

        if self.timedefinition == "AcquisitionTime":
            if "AcquisitionDate" in ds:
                try:
                    acq_date = ds.data_element("AcquisitionDate")
                    assert acq_date is not None
                    date = datetime.datetime.strptime(acq_date.value, "%Y%m%d")
                except Exception:
                    pass

            if "AcquisitionTime" in ds:
                acq_time = ds.data_element("AcquisitionTime")
                try:
                    assert acq_time is not None
                    if re.search(r"\.", acq_time.value):
                        time = datetime.datetime.strptime(acq_time.value, "%H%M%S.%f")
                    else:
                        time = datetime.datetime.strptime(acq_time.value, "%H%M%S")

                except Exception:
                    pass

        elif self.timedefinition == "ContentTime":
            if "ContentDate" in ds:
                content_date = ds.data_element("ContentDate")
                try:
                    assert content_date is not None
                    date = datetime.datetime.strptime(content_date.value, "%Y%m%d")
                except Exception:
                    pass
            if "ContentTime" in ds:
                try:
                    content_time = ds.data_element("ContentTime")
                    assert content_time is not None
                    if re.search(r"\.", content_time.value):
                        time = datetime.datetime.strptime(
                            content_time.value, "%H%M%S.%f"
                        )
                    else:
                        time = datetime.datetime.strptime(content_time.value, "%H%M%S")

                except Exception:
                    pass
        elif self.timedefinition == "TriggerTime":
            trigger_time = ds.data_element("TriggerTime")
            assert trigger_time is not None
            time_s = float(trigger_time.value) / 1000.0
            time = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=time_s)

        else:
            print("Default date and time applied")

        return datetime.datetime.combine(date.date(), time.time())

    def sort(self) -> None:

        sortcolumns = np.zeros((len(self.datasets), 4), int)
        k = 0
        nslicestotal = len(self.datasets)
        slicetimes = []
        for ds, _ in self.datasets:
            if self.is_enhanced:
                raise AssertionError("To be implemented for multiframe!!")
            else:
                IOP = ds.ImageOrientationPatient
                IPP = ds.ImagePositionPatient

            instance_number = ds.InstanceNumber
            z_pos = self.calc_proj_z(IOP, IPP)
            slicetimes.append(self.get_slice_time(ds))
            sortcolumns[k, :] = [z_pos, 1, instance_number, k]
            k = k + 1

        lowesttime = min(slicetimes)
        slicetimes_normalized = [
            (k - lowesttime).total_seconds() * 1000000 for k in slicetimes
        ]

        for k in range(len(slicetimes_normalized)):
            sortcolumns[k, 1] = slicetimes_normalized[k]

        sorted_indx = np.lexsort(
            (sortcolumns[:, 2], sortcolumns[:, 1], sortcolumns[:, 0])
        )

        sortcolumns_sorted = sortcolumns[sorted_indx, :]

        unique_z = np.unique(sortcolumns_sorted[:, 0])

        counts = np.zeros((1, len(unique_z)), int)

        for k, z_pos in enumerate(unique_z):
            counts[0, k] = sum(sortcolumns_sorted[:, 0] == z_pos)

        unique_counts = np.unique(counts)

        if len(unique_counts) == 1:
            self.isophasic = True
            self.nframes = unique_counts[0]
            self.nslices = len(unique_z)
            self.timingmatrix = np.zeros((self.nslices, self.nframes), int)

            self.sortorder = np.zeros((self.nslices, self.nframes), int)

            for islice in range(self.nslices):
                for iframe in range(self.nframes):
                    rowindx = iframe + self.nframes * islice
                    self.sortorder[islice, iframe] = sortcolumns_sorted[rowindx, 3]
                    self.timingmatrix[islice, iframe] = sortcolumns_sorted[rowindx, 1]
        else:
            self.isophasic = False
            self.nframes = -1
            self.nslices = len(unique_z)
            maxframes = unique_counts.max()
            self.timingmatrix = np.zeros((self.nslices, maxframes), int) * np.nan
            self.sortorder = sortcolumns_sorted[:, 3:4]

            maxframes = unique_counts.max()
            expected_frames = maxframes * self.nslices
            print(f"Expected frames {expected_frames}, have {nslicestotal}")

            inumbers = sortcolumns[:, 2]
            exclude_inumbers: list = []
            for iframe in range(maxframes):
                expected_inumbers = np.arange(
                    iframe * self.nslices + 1, (iframe + 1) * self.nslices + 1
                )
                for sl_index, expect in enumerate(expected_inumbers):
                    if expect not in inumbers:
                        print(f"Frame {iframe} not complete")
                        print(
                            f"Instancenumber {expect} is missing - slice index {sl_index}"
                        )
                        expected_inumbers_present = [
                            inumber
                            for inumber in expected_inumbers.tolist()
                            if inumber in inumbers
                        ]
                        exclude_inumbers = exclude_inumbers + expected_inumbers_present
                        break

            print(
                "Quarantine these in order to quarantine frames with excluded i numbers:"
            )
            for inumber in exclude_inumbers:
                row = np.argwhere(inumbers == inumber)[0][0]
                print(self.datasets[row][1])

            expected_inumber_matrix = np.reshape(
                np.arange(1, expected_frames + 1), [self.nslices, maxframes], order="F"
            )
            file_matrix = np.empty((self.nslices, maxframes), dtype=object)

            for islice in range(self.nslices):
                for iframe in range(maxframes):
                    if expected_inumber_matrix[islice, iframe] in inumbers:
                        inumber_row = (
                            sortcolumns_sorted[:, 2]
                            == expected_inumber_matrix[islice, iframe]
                        )
                        dataset_indx = sortcolumns_sorted[inumber_row, 3][0]
                        file_matrix[islice, iframe] = self.datasets[dataset_indx][1]
                    else:
                        file_matrix[islice, iframe] = "NA"

        self.z_pos = unique_z

        second_z_pos_diff = np.unique(np.abs(np.diff(np.diff(unique_z))))
        if np.any(second_z_pos_diff > (self.slicetol * 1000)):
            self.equidistant = 0

        else:
            self.equidistant = 1

        first_ds = self.datasets[self.sortorder[0, 0]][0]
        if "SeriesDescription" in first_ds:
            self.seriesdescription = first_ds.data_element("SeriesDescription").value

        if "Modality" in first_ds:
            self.modality = first_ds.data_element("Modality").value
        else:
            self.modality = "NA"

        self.patientid = first_ds.data_element("PatientID").value

        if "SeriesDate" in first_ds and len(first_ds["SeriesDate"].value) > 0:
            date = first_ds.data_element("SeriesDate").value
        elif self.seriesdate_fallback_tostudy:
            if "StudyDate" in first_ds and len(first_ds["StudyDate"].value) > 0:
                date = first_ds.data_element("StudyDate").value
            else:
                date = None
        else:
            date = None

        if "SeriesTime" in first_ds and len(first_ds["SeriesTime"].value) > 0:
            time = first_ds.data_element("SeriesTime").value
            time = time.replace(":", "")

            if time == "0":
                print("Special cfed fix")
                time = "000000"

            try:
                self.seriesdatetime = datetime.datetime.strptime(
                    date + time[0:6], "%Y%m%d%H%M%S"
                )
            except Exception as e:
                print(
                    f"failure passing date in {self.datasets[self.sortorder[0, 0]][1]}"
                )
                raise (e)
        elif self.seriesdate_fallback_tostudy:
            if "StudyTime" in first_ds and len(first_ds["StudyTime"].value) > 0:
                time = first_ds.data_element("StudyTime").value
                self.seriesdatetime = datetime.datetime.strptime(
                    date + time[0:6], "%Y%m%d%H%M%S"
                )
            else:
                time = None
        else:
            self.seriesdatetime = None

        if "PixelSpacing" in first_ds:
            stepsizes = first_ds.PixelSpacing
            stepsizes.append(np.median(np.diff(unique_z)) / 1000)
            self.stepsizes = np.array(
                [
                    np.float32(stepsizes[0]),
                    np.float32(stepsizes[1]),
                    np.float32(stepsizes[2]),
                ]
            )
        else:
            self.stepsizes = "NA"

        self.issorted = True

    def __str__(self) -> str:
        if not self.issorted:
            return ""

        myop = (
            self.series_instance_uid + ":\n"
            "SeriesDescription: " + self.seriesdescription + "\n"
            "Slices: " + str(self.nslices) + "\n"
            "Frames: " + str(self.nframes) + "\n"
            "Isophasic: " + str(self.isophasic) + "\n"
            "Equidistant: " + str(self.equidistant)
        )

        return myop

    def get_file_order(self) -> List[List[str]]:
        if not self.issorted:
            return [[]]

        filelist = []

        for d1 in range(self.sortorder.shape[0]):
            framelist = []
            for d2 in range(self.sortorder.shape[1]):
                cindx = self.sortorder[d1, d2]
                framelist.append(self.datasets[cindx][1])
            filelist.append(framelist)
        return filelist

    def sitkimage(self, use_rescale: bool = False) -> Tuple[sitk.Image, np.ndarray]:

        files = self.get_file_order()

        img1 = self.datasets[0][0].pixel_array

        if use_rescale:
            datamatrix = np.zeros(
                (img1.shape[0], img1.shape[1], self.nslices, self.nframes), np.float32
            )
        else:
            myformat = img1.dtype
            if img1.dtype == ">u2":
                myformat = "u2"
            datamatrix = np.zeros(
                (img1.shape[0], img1.shape[1], self.nslices, self.nframes), myformat
            )

        for islice in range(self.nslices):
            for iframe in range(self.nframes):
                if use_rescale:
                    if (
                        "RescaleSlope"
                        in self.datasets[self.sortorder[islice, iframe]][0]
                    ):
                        slope = float(
                            self.datasets[self.sortorder[islice, iframe]][
                                0
                            ].RescaleSlope
                        )
                        intercept = float(
                            self.datasets[self.sortorder[islice, iframe]][
                                0
                            ].RescaleIntercept
                        )
                        datamatrix[:, :, islice, iframe] = (
                            self.datasets[self.sortorder[islice, iframe]][0].pixel_array
                            * slope
                            + intercept
                        )
                    else:
                        datamatrix[:, :, islice, iframe] = self.datasets[
                            self.sortorder[islice, iframe]
                        ][0].pixel_array
                else:
                    datamatrix[:, :, islice, iframe] = self.datasets[
                        self.sortorder[islice, iframe]
                    ][0].pixel_array

        flist = [f[0] for f in files]
        reader = sitk.ImageSeriesReader()

        reader.SetFileNames(flist)
        headerimg = reader.Execute()

        if self.nframes > 1:
            datamatrix_reordered = np.transpose(datamatrix, [2, 0, 1, 3])
            dymimg = sitk.GetImageFromArray(datamatrix_reordered, isVector=True)
            dymimg.CopyInformation(headerimg)
            headerimg = dymimg

        return headerimg, datamatrix


def dicomscan(
    inputspec: str | List[str],
    series_grouping_tags: str = "SeriesInstanceUID",
    slicetolin: float = 0.01,
    timedefinition: str = "AcquisitionTime",
    exclude_folder_name_regex: Optional[str] = None,
) -> Dict[str, DicomSeries]:
    foundseries: Dict[str, DicomSeries] = {}
    if type(inputspec) is str:
        rootdir = inputspec

        for folder, _subs, files in os.walk(rootdir):
            if exclude_folder_name_regex is not None:
                if re.search(exclude_folder_name_regex, folder):
                    continue
            for filename in files:
                fname = os.path.join(folder, filename)

                try:
                    ds = pydicom.dcmread(fname)

                except Exception as excp:
                    print(f"Failed on reading {fname}")
                    raise excp

                is_enhanced = False
                sop_classtxt = pdUID(ds["SOPClassUID"].value).name
                if re.match("Enhanced", sop_classtxt):
                    is_enhanced = True

                UID = ""
                for k in series_grouping_tags.split(","):
                    if k in ds:
                        c_data_element = ds.data_element(k)
                        assert c_data_element is not None
                        UID = UID + str(c_data_element.value) + ","
                    else:
                        assert 0
                UID = UID[0:-1]

                if UID not in foundseries:
                    foundseries[UID] = DicomSeries(
                        UID,
                        slicetol=slicetolin,
                        timedefinition=timedefinition,
                        is_enhanced=is_enhanced,
                    )
                foundseries[UID].add_ds((ds, fname))

    elif type(inputspec) is list:
        for fname in inputspec:
            try:
                ds = pydicom.dcmread(fname)
            except Exception:
                print(f"Ignoring non-dicom file: {fname}")
                continue

            UID = ""
            for k in series_grouping_tags.split(","):
                if k in ds:
                    c_data_element = ds.data_element(k)
                    assert c_data_element is not None
                    UID = UID + str(c_data_element.value) + ","
                else:
                    assert 0
            UID = UID[0:-1]

            if UID not in foundseries:
                foundseries[UID] = DicomSeries(
                    UID, slicetol=slicetolin, timedefinition=timedefinition
                )

            foundseries[UID].add_ds((ds, fname))

    for series_uid in foundseries.keys():
        foundseries[series_uid].sort()

    return foundseries


def write_ct_dicom(fileloc: Path, matrix: np.ndarray, header: Dict[str, Any]) -> None:

    sop_instance_uid = pydicom.uid.generate_uid()
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    file_meta.FileMetaInformationGroupLength = 10
    file_meta.FileMetaInformationGroupLength = file_meta.__sizeof__()
    ds = pydicom.dataset.FileDataset(
        "dummy.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128
    )
    ds.is_implicit_VR = False

    ds.Modality = "CT"
    ds.PatientName = header["PatientName"]
    ds.PatientID = header["PatientID"]
    ds.PatientBirthDate = header["PatientBirthDate"]
    ds.PatientSex = header["PatientSex"]

    ds.StudyInstanceUID = header["StudyInstanceUID"]
    ds.SeriesDate = header["SeriesDate"]
    ds.SeriesTime = header["SeriesTime"]
    ds.StudyDate = header["StudyDate"]
    ds.StudyTime = header["StudyTime"]
    ds.ReferringPhysicianName = "Dr NonCon"

    ds.StudyID = ""
    ds.AccessionNumber = ""

    if "StudyID" in header:
        ds.StudyID = header["StudyID"]

    if "AccessionNumber" in header:
        ds.AccessionNumber = header["AccessionNumber"]

    ds.StudyDescription = header["StudyDescription"]

    ds.SeriesDescription = header["SeriesDescription"]

    ds.SeriesInstanceUID = header["SeriesInstanceUID"]
    ds.SeriesNumber = header["SeriesNumber"]
    ds.PatientPosition = "HFS"

    ds.FrameOfReferenceUID = header["FrameOfReferenceUID"]
    ds.PositionReferenceIndicator = ""

    ds.Manufacturer = "StrokeCenter"

    ds.InstanceNumber = header["InstanceNumber"]
    if "AcquisitionDate" in header:
        ds.AcquisitionDate = header["AcquisitionDate"]

    if "AcquisitionTime" in header:
        ds.AcquisitionTime = header["AcquisitionTime"]

    ds.PixelSpacing = header["PixelSpacing"]
    ds.ImageOrientationPatient = header["ImageOrientationPatient"]

    ds.ImagePositionPatient = header["ImagePositionPatient"]
    ds.SliceThickness = ""

    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = header["Rows"]
    ds.Columns = header["Columns"]
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    ds.PixelData = matrix.astype("<i2").tobytes()

    ds.ImageType = "ORIGINAL"
    ds.RescaleType = "HU"
    ds.RescaleIntercept = header["RescaleIntercept"]
    ds.RescaleSlope = header["RescaleSlope"]
    ds.KVP = ""

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"

    ds.SOPInstanceUID = sop_instance_uid

    ds.save_as(str(fileloc))


def sitk2generic_ct(
    ct: sitk.Image,
    ct_outfolder: Path,
    seriesheader: Dict[str, Any] | None = None,
    instance_number_start: int = 0,
) -> None:
    ct_outfolder.mkdir(parents=True, exist_ok=True)

    ctarr = sitk.GetArrayViewFromImage(ct)
    spacing = ct.GetSpacing()
    header = {
        "SeriesInstanceUID": pydicom.uid.generate_uid(),
        "StudyInstanceUID": pydicom.uid.generate_uid(),
        "FrameOfReferenceUID": pydicom.uid.generate_uid(),
        "Rows": ctarr.shape[1],
        "Columns": ctarr.shape[2],
        "PixelSpacing": [
            "{:2.8f}".format(spacing[0]),
            "{:2.8f}".format(spacing[1]),
        ],
        "PatientID": "Noname",
        "PatientBirthDate": "19000101",
        "PatientSex": "F",
        "PatientName": "Noname",
        "StudyDescription": "IndescriptStudy",
        "SeriesDescription": "IndescriptSeries",
        "StudyDate": datetime.datetime.now().strftime("%Y%m%d"),
        "StudyTime": datetime.datetime.now().strftime("%H%M%S"),
        "SeriesDate": datetime.datetime.now().strftime("%Y%m%d"),
        "SeriesTime": datetime.datetime.now().strftime("%H%M%S"),
        "Modality": "CT",
        "RescaleSlope": 1,
        "RescaleIntercept": -1024,
        "SeriesNumber": "100",
    }

    if seriesheader is None:
        seriesheader = {}

    header_use = header | seriesheader

    IOP = (
        np.array(ct.GetDirection())
        .reshape([3, 3], order="C")
        .reshape(-1, order="F")
        .tolist()[0:6]
    )

    nslices = ctarr.shape[0]

    for k in range(nslices):
        IPP = ct[:, :, k : (k + 1)].GetOrigin()

        instancenumber = k + instance_number_start
        SOPInstanceUID = pydicom.uid.generate_uid()
        header_use["SOPInstanceUID"] = SOPInstanceUID
        header_use["InstanceNumber"] = instancenumber
        header_use["ImagePositionPatient"] = [f"{v:2.5}" for v in IPP]
        header_use["ImageOrientationPatient"] = IOP

        write_ct_dicom(
            ct_outfolder / (SOPInstanceUID + ".dcm"),
            ctarr[k, :, :] - header["RescaleIntercept"],
            header_use,
        )


if __name__ == "__main__":
    d = "/home/sorenc/DATA/AImedical/lukaku/incoming/coregProducts_A910789F-45AE-4589-803F-E01B6EBD52F3/002_BrainVsBrainNeck/fixImage"
    op = dicomscan(d)
