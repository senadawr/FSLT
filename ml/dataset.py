from dataclasses import asdict, dataclass
from pathlib import Path
import json
from collections import defaultdict


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    signer_id: str | None
    session_id: str | None
    sign_label: str
    gloss_label: str | None
    source_dataset: str
    sequence_length: int
    feature_version: str
    file_path: str


def write_metadata(records: list[SampleMetadata], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )


def split_by_signer(
    records: list[SampleMetadata],
    validation_signers: set[str],
    test_signers: set[str],
) -> dict[str, list[SampleMetadata]]:
    """Split without leaking samples from known signers across partitions."""
    overlap = validation_signers & test_signers
    if overlap:
        raise ValueError(f"Signer appears in validation and test sets: {sorted(overlap)}")
    partitions: dict[str, list[SampleMetadata]] = defaultdict(list)
    for record in records:
        if record.signer_id is None:
            partitions["unassigned"].append(record)
        elif record.signer_id in test_signers:
            partitions["test"].append(record)
        elif record.signer_id in validation_signers:
            partitions["validation"].append(record)
        else:
            partitions["train"].append(record)
    return dict(partitions)
