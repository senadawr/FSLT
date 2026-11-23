# Dataset preparation

## WLASL 2000 representative frames

The processed Kaggle mirror contains 21,083 PNG files across 2,000 gloss folders.
Each PNG is a single representative frame, not a temporal video sequence. It
provides WLASL metadata with train/val/test splits and signer IDs.

`ml.wlasl_landmarks.py` extracts 42 left/right hand landmarks from each PNG,
normalizes each detected hand independently, and writes split arrays. The
license is C-UDA-1.0 and permits computational use under the dataset terms,
not automatic commercial use.

This is static isolated-sign data. It does not support movement, velocity,
continuous signing, or temporal recognition.

## Whole-word ASL skeleton videos

A public Kaggle word dataset provides 2,000 isolated ASL words as MediaPipe
skeletal MP4s. It is MIT licensed and includes `words.txt`; the files are:

- 2,031 MP4 files
- 2,000 word labels in `words.txt`
- 26 alphabet videos outside the word list
- 640×480, approximately 25–33 FPS
- No signer IDs or published train/validation/test metadata

Use `ml.asl_words_skeletal.py` to verify labels and extract frame sequences.
This is isolated word recognition, not continuous ASL or complete translation.

## ASL Alphabet Hand Landmarks

The public ASL Alphabet Hand Landmarks dataset is directly compatible with the
21-point finger skeleton:

- 10,508 samples
- 26 classes (A–Z)
- MediaPipe 21-point 3D landmarks
- License: CC0 Public Domain
- No signer IDs are included

Download it with the authenticated Kaggle CLI:

```powershell
.\.venv\Scripts\python.exe -m kaggle datasets download -d borisgraudt/asl-alphabet-hand-landmarks -p data/asl-alphabet
Expand-Archive data/asl-alphabet/asl-alphabet-hand-landmarks.zip -DestinationPath data/asl-alphabet
.\.venv\Scripts\python.exe ml/asl_landmarks.py --dataset-dir data/asl-alphabet
```

The importer expects `landmarks/<A-Z>/*.npy`, validates `(21, 3)` shapes and
finite values, wrist-normalizes coordinates, and creates
train/validation/test splits. It explicitly reports that signer-independent
evaluation is unavailable because the dataset has no signer IDs.

The FSL-105 importer is available as `ml.fsl105.import_fsl105`. It reads the
published `labels.csv`, `train.csv`, and `test.csv` files and verifies that the
row labels agree with `labels.csv`. Run it against the extracted dataset
directory; pass `check_video_files=True` (the default) to report missing video
files.

The published FSL-105 version 2 metadata was inspected locally. It contains
2,130 rows, 105 labels, and Windows-style paths such as
`clips\17\6.MOV`. The CSVs provide no signer or session identifiers, so this
dataset cannot support an unseen-signer evaluation claim.

Importers must inspect the downloaded dataset manifest before reading samples.
Each output record uses `SampleMetadata` and records unavailable signer or
session identifiers as `null`; identities must never be invented.

The FSL-105 and Transactional Filipino Sign Language datasets should remain
separate until their labels, feature order, licensing, and sample structure
have been verified. Split by signer when signer IDs exist. If they do not,
report that unseen-signer evaluation is unavailable.

## Dataset A download

Dataset A is FSL-105 version 2:
<https://data.mendeley.com/datasets/48y2y99mb9/2>. Its metadata files are
small and can be downloaded from the page; `clips.zip` is approximately 379 MB.
The dataset is licensed CC BY 4.0. Do not commit the downloaded archive or
video files.
