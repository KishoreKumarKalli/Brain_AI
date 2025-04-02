# Brain Segmentation and Analysis Framework

## Project Overview

This framework provides an end-to-end solution for brain MRI analysis, including segmentation, abnormality detection, and statistical analysis. It leverages advanced deep learning models and statistical methods to process T1-weighted MRI scans and correlate imaging findings with clinical data.

### Key Features

- **Brain Segmentation:** Segment anatomical regions of the brain (gray matter, white matter, ventricles, hippocampus, etc.)
- **Abnormality Detection:** Identify potential abnormalities such as tumors, lesions, and atrophy
- **Statistical Analysis:** Perform quantitative analysis comparing subject groups and correlating with clinical measures
- **Web Interface:** Simple Flask-based interface for uploading scans and viewing results

## Dataset

This project uses T1-weighted MRI scans from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset:
- 40 CN (Cognitively Normal) subjects
- 40 MCI (Mild Cognitive Impairment) subjects
- 20 AD (Alzheimer's Disease) subjects

Clinical data is sourced from the following ADNI files:
- ADNI_T1.csv (Main subject information)
- CDR.csv (Clinical Dementia Rating)
- MMSE.csv (Mini-Mental State Examination)
- GDSCALE.csv (Geriatric Depression Scale)
- ADAS_ADNI1.csv (Alzheimer's Disease Assessment Scale for ADNI1)
- ADAS_ADNIGO23.csv (ADAS for ADNI2, ADNI3, and ADNIGO)
- NEUROBAT.csv (Neuropsychological Battery)
- PTDEMOG.csv (Patient Demographics)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)


Then navigate to `http://localhost:5000` in your web browser.

## Pipeline Components

### 1. Data Preprocessing

- NIfTI image loading and standardization
- Intensity normalization
- Skull stripping (optional, uses pre-trained models)
- Spatial normalization to standard space

### 2. Brain Segmentation

Leverages MONAI/nnUNet pre-trained models to segment:
- Gray matter
- White matter
- Cerebrospinal fluid
- Subcortical structures
- Ventricles
- Hippocampus
- Other anatomical regions

### 3. Abnormality Detection

- Identifies potential abnormalities using deep learning models
- Highlights regions with potential pathologies
- Quantifies deviations from normal appearance

### 4. Statistical Analysis

- Group comparisons (CN vs. MCI vs. AD)
- Correlation with clinical measures (MMSE, CDR, etc.)
- Volume quantification of segmented regions
- Statistical significance testing
- Visualization of results


## Acknowledgments

- ADNI for providing the neuroimaging and clinical data
- MONAI for open-source medical imaging AI framework
- The open-source neuroimaging community
