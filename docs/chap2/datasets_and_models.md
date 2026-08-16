---
layout: default
title: Datasets and Models
parent: CHAP2.0
nav_order: 1
---

# CHAP2.0 Datasets and Models
{: .no_toc }

This page summarizes the datasets and checkpoints used in the CHAP2.0
workflows. For preprocessing commands and input arguments, see
[CHAP2.0 Data Preprocessing]({{ site.baseurl }}{% link chap2/preprocess.md %}).

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Datasets

### About

#### SOL/PASOS
SOL PASOS (Peripheral Artery Disease Study of SOL) is an ancillary study to the NHLBI-sponsored Hispanic Community Health Study/Study of Latinos (HCHS/SOL), a population-based prospective cohort study of 4,370 Hispanic/Latino adults and older adults aged 40 to 80 years in four regions of the U.S. SOL PASOS was conducted with HCHS/SOL Visit 3 in 2020-2024 and included up to 8 days of 24-hour measurement using wrist-worn ActiGraph monitors, with a subsample of 621 participants also wearing an activPAL. The activPAL measured sitting/lying and standing, and time in bed was excluded using participant bed logs and activPAL in-bed detection. The ActiGraph recorded raw triaxial acceleration in gravitational units, primarily at 80 Hz. The study was approved by the Institutional Review Board (IRB) at the University of North Carolina, and all participants provided informed consent.

#### iWatch
The iWatch study was an observational study of sitting and physical activity patterns in 148 adults aged 24 to 85 years, of whom 50% were female, 22% were non-White, and 20% were Hispanic. Daily activities were measured using a person-worn SenseCam (Vicon Revue), which captured first-person images approximately every 20 seconds while participants wore the camera on a lanyard for 7 days, excluding nighttime charging periods. SenseCam annotations were collapsed into two CHAP2.0 classes: Sitting, including Sedentary/Sitting and Vehicle, and Nonsitting, including Standing Still, Standing Moving, and Walking/Running. Nonwear and uncodable intervals were excluded. Data collection also included hip- and wrist-worn ActiGraph GT3X+ accelerometers, worn at the hip/waist or on the non-dominant wrist, with raw triaxial acceleration recorded in gravitational units at 30 Hz. The study was approved by the Institutional Review Board (IRB) at the University of California, San Diego, IRB #111160, date 07/12/2011, and all participants provided informed consent.

### Summary

| Dataset | Devices | Wear location | Label source | Sampling frequency |
|---------|---------|---------------|--------------|--------------------|
| SOL/PASOS | ActiGraph and activPAL | Wrist ActiGraph; activPAL subsample | activPAL sitting/lying and standing labels | ActiGraph primarily 80 Hz |
| iWatch | ActiGraph GT3X+ and SenseCam | Hip and wrist ActiGraph | SenseCam image annotations collapsed to sitting and nonsitting | ActiGraph 30 Hz |

## Data Format

CHAP2.0 preprocessing converts raw ActiGraph CSV or CSV.GZ files and
dataset-specific annotation files into daily HDF5 files.

### SOL/PASOS input files

For SOL/PASOS, preprocessing expects wrist ActiGraph raw files and can use
supporting annotation files for valid wear days, sleep or in-bed time,
non-wear intervals, and activPAL posture labels.

| Input | Description |
|-------|-------------|
| `--gt3x-dir` | Directory containing SOL/PASOS ActiGraph raw CSV files. Use the matching `--gt3x-frequency` for the input files. |
| `--valid-days-file` | Optional CSV identifying concurrent wear or valid analysis days. |
| `--sleep-logs-file` | Optional sleep or in-bed interval file. SOL/PASOS sleep logs use subject ID plus sleep start and sleep end timestamps. |
| `--non-wear-times-file` | Optional non-wear interval file. SOL/PASOS non-wear files use subject ID plus non-wear start and end timestamps. |
| `--activpal-dir` | Directory containing activPAL label files. Leave `--event-file` unset for 1-second epoch files with `TS_LOCAL_COR` and `PL_ACTIVITY_NEW` columns. |

The ActiGraph raw files are gzipped CSV files. The first few lines of a sample
file are as follows:

```text
------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.5 Firmware v1.7.2 date format M/d/yyyy at 80 Hz  Filter Normal -----------
Serial Number: TAS1H45190074
Start Time 09:06:00
Start Date 8/18/2023
Epoch Period (hh:mm:ss) 00:00:00
Download Time 15:16:26
Download Date 8/30/2023
Current Memory Address: 0
Current Battery Voltage: 3.75     Mode = 12
--------------------------------------------------
Accelerometer X,Accelerometer Y,Accelerometer Z
-0.227,0.977,-0.059
-0.207,0.961,-0.051
```

The activPAL files provide 1-second epoch posture labels:

```text
HCHSID,TS_LOCAL_COR,TS_LOCAL,TS_UTC,PL_ACTIVITY,PL_ACTIVITY_NEW,PL_DUR,PL_WAKE_PRD,PL_STEPS,PL_MET_H,PL_WAKESLEEPDAY_ID
M7148383,2022-04-19T11:20:49Z,2022-04-19T11:20:49Z,2022-04-19T15:20:49Z,0,0,10546.3,1,0,0.000347222222222222,1
M7148383,2022-04-19T11:20:50Z,2022-04-19T11:20:50Z,2022-04-19T15:20:50Z,0,0,10546.3,1,0,0.000347222222222222,1
```

The valid-days file identifies subject days with concurrent wear:

```text
"ID","valid_days"
"C6002331",2023-11-25
"C6002331",2023-11-26
"C6002331",2023-11-27
```

The sleep-log file contains subject-level sleep or in-bed intervals:

```text
ID,startsleep,endsleep
B5025951,3/28/23 23:00,3/29/23 9:09
B5025951,3/30/23 20:29,3/31/23 9:14
```

The non-wear file contains subject-level non-wear intervals:

```text
ID,startNW,endNW
C6003361,6/20/22 5:39,6/20/22 7:16
C6003361,6/20/22 23:58,6/21/22 10:12
```

For datasets with mixed raw sampling frequencies, split raw files by frequency
and run preprocessing separately with the matching `--gt3x-frequency`. For
example, if a SOL/PASOS input directory contains both 60 Hz and 80 Hz files,
process those subject groups separately rather than relying on one mixed run.

### iWatch input files

For iWatch, preprocessing expects hip or wrist ActiGraph raw files,
SenseCam-derived event labels, and support files that identify concurrent wear
days, non-wear intervals, and SenseCam wear periods. Use the hip-specific
concurrent wear file for hip ActiGraph data and the wrist-specific concurrent
wear file for wrist ActiGraph data.

| Input | Description |
|-------|-------------|
| `--gt3x-dir` | Directory containing iWatch ActiGraph raw CSV or CSV.GZ files. Use `--gzipped` for `.csv.gz` files. Hip and wrist files are stored separately. |
| `--activpal-dir` | Directory containing SenseCam-derived event label CSV files matched to the raw ActiGraph file names. Use `--event-file` for this format. |
| `--valid-days-file` | Concurrent wear day CSV. Use the hip file for hip ActiGraph data and the wrist file for wrist ActiGraph data. |
| `--non-wear-times-file` | iWatch ActiGraph non-wear interval file. If the file includes both hip and wrist records, pass `--loc hip` or `--loc wrist`. |
| `--wear-logs-file` | SenseCam-derived wear interval CSV. The preprocessing script uses it to anchor observed wear periods when sleep logs are not available. |

The ActiGraph raw files are gzipped CSV files from hip (`H`) or wrist (`W`)
wear. Wrist files are from non-dominant wrist wear or wrist wear where the hand
is unknown; known dominant-wrist records were excluded. The first few lines of a
sample file are as follows:

```text
------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.4 Firmware v3.2.1 date format M/d/yyyy at 30 Hz  Filter Normal -----------
Serial Number: MRA1F08120057
Start Time 00:00:00
Start Date 4/16/2014
Epoch Period (hh:mm:ss) 00:00:00
Download Time 13:16:21
Download Date 4/29/2014
Current Memory Address: 0
Current Battery Voltage: 3.85     Mode = 12
--------------------------------------------------
Accelerometer X,Accelerometer Y,Accelerometer Z
-0.393,0.094,0.962
-0.387,0.094,0.956
```

The event label files come from SenseCam annotations formatted as event-style
activPAL input. These files are the ground-truth labels and are shared for hip
and wrist ActiGraph preprocessing:

```text
"Time","DataCount (samples)","Interval (s)","ActivityCode (0=sedentary, 1= standing, 2=stepping)","CumulativeStepCount","Activity Score (MET.h)","Abs(sumDiff)"
"41820.3555555556","",120,1,"","",""
"41820.3569444444","",780,0,"","",""
"41820.3833333333","",360,-1,"","",""
"41820.3881944444","",120,-2,"","",""
```

The concurrent wear files identify valid subject days when the required devices
were worn together. The preprocessing script marks days outside this list as
non-wear:

```text
"ID","valid_days","wearloc"
"i0001A",2013-05-07,"H"
"i0001A",2013-05-08,"H"
"i0001A",2013-05-09,"H"
"i0001A",2013-05-10,"H"
```

The iWatch non-wear file contains ActiGraph non-wear intervals detected from the
accelerometer data. It can include both hip and wrist records, so use `--loc` to
select the wear location:

```text
"ID","wearLoc","NW_DT","int.min","wearDate","timeNum","filename","loc"
"i0234A","N",2014-04-16,1440,2014-04-16,0,"i0234A_N2005760sec.agd","Wrist"
"i0234A","N",2014-04-17,874,2014-04-17,0,"i0234A_N2005760sec.agd","Wrist"
"i0234A","N",2014-04-17 15:40:00,500,2014-04-17,940,"i0234A_N2005760sec.agd","Wrist"
```

The wear-log file is derived from SenseCam wear periods and is used to anchor
when participants were observed to be wearing the camera:

```text
"shortID","startWear","endWear","wearStart","wearEnd","wearDate","timeNum","int.min"
"i0001A",2013-05-07 14:46:00,2013-05-07 22:42:00,"2013-05-07","2013-05-07","2013-05-07",886,476
"i0001A",2013-05-08 07:02:00,2013-05-08 07:09:00,"2013-05-08","2013-05-08","2013-05-08",422,7
"i0001A",2013-05-08 07:17:00,2013-05-08 07:49:00,"2013-05-08","2013-05-08","2013-05-08",437,32
```

The preprocessing script converts these inputs into daily HDF5 files used by
CHAP2.0 prediction and fine-tuning. See
[Data Preprocessing]({{ site.baseurl }}{% link chap2/preprocess.md %}) for the
output HDF5 structure.

## Recommended Checkpoints

Use these checkpoints as the default recommendations for CHAP2.0 prediction:

| Checkpoint | Recommended for |
|------------|-----------------|
| `SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth` | ActiGraph wrist data. |
| `SUBMIT_RESULT/iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth` | ActiGraph hip data (ACT and AusDiab trained). |

The SOL/PASOS fine-tuned wrist checkpoint is the recommended wrist model. The
iWatch wrist checkpoint is retained for reproducibility, but is not the default
recommendation.

## Submitted Checkpoints

CHAP2.0 submitted checkpoints are stored under `CHAP2/SUBMIT_RESULT/` and can
be used with the CHAP2.0
[prediction workflow]({{ site.baseurl }}{% link chap2/prediction.md %}).

CHAP-ZS means zero-shot prediction: the CHAP/MSSE-2021 pre-trained checkpoint
is applied directly to the target dataset without fine-tuning. CHAP-FT
checkpoints are initialized from CHAP/MSSE-2021 weights and then fine-tuned on
the target dataset.

The following checkpoints are included for reproducibility and comparison; not
all are recommended as default models.

| Checkpoint | Type | Target data | Notes |
|------------|------|-------------|-------|
| `SOL_W/CHAP_FT/checkpoint-submit.pth` | CHAP-FT | SOL/PASOS wrist | Fine-tuned on SOL/PASOS wrist data; recommended for ActiGraph wrist data. |
| `iWatch_W/CHAP-FT/checkpoint/checkpoint-submit.pth` | CHAP-FT | iWatch wrist | Fine-tuned on iWatch wrist data; retained for reproducibility. |
| `iWatch_W/CHAP-ZS/checkpoint/checkpoint-submit.pth` | CHAP-ZS | iWatch wrist | Zero-shot baseline for iWatch wrist data. |
| `iWatch_H/CHAP-FT/checkpoint/checkpoint-submit.pth` | CHAP-FT | iWatch hip | Fine-tuned on iWatch hip data; retained for reproducibility. |
| `iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth` | CHAP-ZS | iWatch hip | Recommended checkpoint for ActiGraph hip data trained on ACT and AusDiab. |
