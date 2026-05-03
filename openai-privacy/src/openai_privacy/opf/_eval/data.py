import glob
import gzip
import io
import json
import os
import re
from typing import Iterable, Iterator, Mapping, Sequence


MESSAGE_PREFIX_RE = re.compile(r"(?:^|\n\n)(user|assistant): ")


def _split_message_spans(text: str) -> list[tuple[int, int]]:
    """Split a chat transcript into message-body character spans."""
    matches = list(MESSAGE_PREFIX_RE.finditer(text))
    if not matches:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        spans.append((start, end))
    return spans


def iter_message_variant_records(
    records: Iterable[Mapping[str, object]],
) -> Iterator[Mapping[str, object]]:
    """Yield message-level eval records derived from full transcript records."""
    for record in records:
        text = str(record.get("text", ""))
        spans = record.get("spans") or {}
        for message_index, (start, end) in enumerate(_split_message_spans(text)):
            if start >= end:
                continue
            message_text = text[start:end]
            message_record = dict(record)
            message_record["text"] = message_text
            if isinstance(spans, Mapping):
                adjusted_spans: dict[str, list[list[int]]] = {}
                for label_key, offsets in spans.items():
                    if not isinstance(offsets, Sequence) or isinstance(
                        offsets, (bytes, str)
                    ):
                        continue
                    new_offsets: list[list[int]] = []
                    for span in offsets:
                        if (
                            not isinstance(span, Sequence)
                            or isinstance(span, (bytes, str))
                            or len(span) != 2
                        ):
                            continue
                        span_start, span_end = int(span[0]), int(span[1])
                        if span_start < start or span_end > end:
                            continue
                        new_offsets.append([span_start - start, span_end - start])
                    if new_offsets:
                        adjusted_spans[str(label_key)] = new_offsets
                message_record["spans"] = adjusted_spans
            info_field = message_record.get("info")
            if isinstance(info_field, Mapping):
                info_mapping = dict(info_field)
            else:
                info_mapping = {}
            info_mapping["message_index"] = message_index
            message_record["info"] = info_mapping
            yield message_record


def iter_json_records(dataset_path: str) -> Iterator[Mapping[str, object]]:
    """Yield eval records from local JSON/JSONL(.gz) files or globs."""
    file_paths: list[str]
    if "://" in dataset_path:
        raise ValueError("Dataset path must be local filesystem only (no URI schemes).")
    if "*" in dataset_path:
        file_paths = sorted(glob.glob(dataset_path))
    elif os.path.exists(dataset_path):
        file_paths = [dataset_path]
    else:
        file_paths = sorted(glob.glob(f"{dataset_path.rstrip('/')}/*.gz"))

    if not file_paths:
        raise FileNotFoundError(f"No gzipped JSONL shards matched {dataset_path!r}")

    for file_path in file_paths:
        with open(file_path, "rb") as raw_file:
            if file_path.endswith(".gz"):
                gz_file = gzip.GzipFile(fileobj=raw_file)
                text_stream = io.TextIOWrapper(gz_file, encoding="utf-8")
            else:
                text_stream = io.TextIOWrapper(raw_file, encoding="utf-8")
            with text_stream as handle:
                for line_idx, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {file_path} line {line_idx}: {exc}"
                        ) from exc
                    if not isinstance(record, Mapping):
                        raise ValueError(
                            f"Golden record in {file_path} line {line_idx} is not an object"
                        )
                    yield record
