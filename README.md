# Medical Imaging Final Assignment

## Introduction

This project focuses on DICOM loading, visualization, and 3D rigid coregistration of medical images. The goal is to provide a comprehensive understanding of the dataset and perform advanced image analysis techniques. The following sections outline the steps involved in the project and provide instructions for executing each task.

## DICOM Loading and Visualization

**a) Dataset Download**: Begin by downloading the dataset HCC-TACE-Seg (click [here](dataset_link)). In the following, consider only the patient HCC XYZ assigned to you (see Aula Digital).

**b) DICOM Visualization**: To visualize the DICOM files, we recommend using a third-party DICOM visualizer such as 3D-Slicer. Open the DICOM files in the visualizer to explore the medical images.

**c) Loading and Rearranging Images**: Use the PyDicom library to load the segmentation image and the corresponding CT image. Rearrange the 'pixel array' of the images based on the relevant headers such as 'Acquisition Number', 'Slice Index', 'Per-frame Functional Groups Sequence - Image Position Patient', and 'Segment Identification Sequence - Referenced Segment Number'.

**d) Animation Creation**: Create an animation, such as a GIF file, with a rotating Maximum Intensity Projection on the coronal-sagittal planes. This will provide a dynamic visualization of the medical images.

## 3D Rigid Coregistration

**a) Image Coregistration**: Perform the coregistration of the given images using either landmarks (defined by you) or a function similarity measure (implemented by you). Avoid using external libraries like PyElastix. The reference image for coregistration is the 'icbm avg 152 t1 tal nlin symmetric VI,' a T1 RM phantom in a normalized space. The input image is the 'RM Brain 3D-SPGR' of an anonymized patient.

**b) Thalamus Visualization**: Visualize the Thalamus region on the input image space to identify the changes after coregistration.

## File structure
- The `HCC_011` contains the DICOM files for the project.
- The `coregistration` folder contains the files for the 3D rigid coregistration.
- The `dicom.py` file is used for the DICOM part.
- The `coregistration.py` file is used for the 3D rigid coregistration part.
