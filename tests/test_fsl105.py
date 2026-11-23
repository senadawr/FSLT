from pathlib import Path

from ml.fsl105 import import_fsl105


def test_imports_published_fsl105_metadata():
    dataset_dir = Path(__file__).parents[1] / "data" / "fsl-105"
    imported = import_fsl105(dataset_dir, check_video_files=False)
    assert len(imported.records) == 2130
    assert imported.label_count == 105
    assert imported.signer_ids_available is False
    assert imported.records[0].source_dataset == "FSL-105"
