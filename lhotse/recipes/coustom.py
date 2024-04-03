import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

# corpus_dir = "/home/dataset/data/lo-datasets/"
# output_dir = "/home/dataset/data/"


def download_coustom(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "http://www.openslr.org/resources",
) -> Path:
    logging.warning(
        "Skipping download!!! Custom datasets are not supported for download!")
    return None


def text_normalize(line: str):
    # 文本预处理
    line = line.upper()
    return line


def prepare_coustom(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset_parts = ["train", "dev", "test"]
    manifests = defaultdict(dict)

    for part in tqdm(dataset_parts, desc="Process audio"):
        recordings = []
        supervisions = []
        csv_path = corpus_dir / f"{part}.tsv"
        with open(csv_path, "r", encoding='utf-8') as f:
            for line in f.readlines():
                audio_path, transcript, speaker = line.split("\t")
                idx = audio_path.split("/")[-1].replace(".wav", "")
                speaker = speaker.strip()
                transcript = text_normalize(transcript)

                recording = Recording.from_file(audio_path)
                recordings.append(recording)

                segment = SupervisionSegment(
                    id=idx,
                    recording_id=idx,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="lao",
                    speaker=speaker,
                    text=transcript.strip().replace(" ", ""),
                )
                supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(
            recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"coustom_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"coustom_recordings_{part}.jsonl.gz")
        manifests[part] = {"recordings": recording_set,
                           "supervisions": supervision_set}
    return manifests

