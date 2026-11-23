import csv
from dataclasses import dataclass
from pathlib import Path

from .dataset import SampleMetadata


@dataclass(frozen=True)
class FSL105Import:
    records: list[SampleMetadata]
    label_count: int
    missing_video_count: int
    signer_ids_available: bool


def import_fsl105(dataset_dir: Path, check_video_files: bool = True) -> FSL105Import:
    """Read the published FSL-105 CSV metadata without inventing signer data."""
    labels = _read_labels(dataset_dir / "labels.csv")
    records: list[SampleMetadata] = []
    missing_video_count = 0
    for split in ("train", "test"):
        with (dataset_dir / f"{split}.csv").open(newline="", encoding="utf-8-sig") as file:
            for row_number, row in enumerate(csv.DictReader(file), start=2):
                video_path = Path(row["vid_path"].replace("\\", "/"))
                label_id = int(row["id_label"])
                expected_label = labels.get(label_id)
                if expected_label is None:
                    raise ValueError(f"{split}.csv:{row_number} references unknown label id {label_id}")
                if row["label"] != expected_label:
                    raise ValueError(f"{split}.csv:{row_number} label does not match labels.csv")
                resolved_video = dataset_dir / video_path
                if check_video_files and not resolved_video.is_file():
                    missing_video_count += 1
                records.append(SampleMetadata(
                    sample_id=f"fsl105-{split}-{row_number - 1}",
                    signer_id=None,
                    session_id=None,
                    sign_label=row["label"],
                    gloss_label=row["label"],
                    source_dataset="FSL-105",
                    sequence_length=0,
                    feature_version="unprocessed",
                    file_path=str(video_path),
                ))
    return FSL105Import(records, len(labels), missing_video_count, signer_ids_available=False)


def _read_labels(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        return {int(row["id"]): row["label"] for row in rows}
