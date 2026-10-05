import os

import torch
from torch.utils.data import DataLoader

from diffsynth.core.data import UnifiedDataset


def test_cached_dataset_order_and_seeded_shuffle_are_reproducible(
    tmp_path, monkeypatch
):
    for name, value in [("z.pth", 1), ("a/second.pth", 2), ("a/first.pth", 3)]:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        torch.save({"value": value}, path)
    (tmp_path / "ignored.txt").write_text("not a cache")
    original_listdir = os.listdir

    def collect(reverse):
        monkeypatch.setattr(
            os, "listdir", lambda path: sorted(original_listdir(path), reverse=reverse)
        )
        dataset = UnifiedDataset(base_path=str(tmp_path), repeat=2)
        loader = DataLoader(
            dataset, shuffle=True, generator=torch.Generator().manual_seed(42)
        )
        return dataset, [batch["value"].item() for batch in loader]

    forward, forward_samples = collect(False)
    backward, backward_samples = collect(True)
    assert forward_samples == backward_samples
    assert forward.cached_data == backward.cached_data
    assert len(forward) == len(backward) == 6
    assert sorted(forward_samples) == [1, 1, 2, 2, 3, 3]


def test_metadata_order_is_preserved(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('[{"value": 2}, {"value": 1}]')
    dataset = UnifiedDataset(base_path=str(tmp_path), metadata_path=str(metadata))
    assert [dataset[index]["value"] for index in range(len(dataset))] == [2, 1]
