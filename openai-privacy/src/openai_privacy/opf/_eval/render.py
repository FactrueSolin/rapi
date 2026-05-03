import codecs
from typing import Sequence

import tiktoken
from .._common.label_space import BACKGROUND_CLASS_LABEL
from .._common.terminal_colors import build_label_color_map, style_labeled_text


def build_prediction_preview(
    *,
    example_id: str,
    text: str,
    token_ids: Sequence[int],
    predicted_spans: Sequence[tuple[int, int, int]],
    span_class_names: Sequence[str],
    encoding: tiktoken.Encoding,
    max_tokens: int,
    max_chars: int,
) -> str:
    """Build the ANSI preview block shown by ``opf eval --preview``."""
    decoded_text, token_fragments, char_starts, char_ends = _decode_tokens(
        token_ids, encoding
    )
    source_text = text
    mismatch_warning = None
    if text != decoded_text:
        source_text = decoded_text
        mismatch_warning = "preview.warning: input text mismatched decoded tokens; using decoded token text."

    normalized_spans = _normalize_predicted_spans(predicted_spans, len(token_ids))
    span_views = _build_span_views(
        normalized_spans=normalized_spans,
        char_starts=char_starts,
        char_ends=char_ends,
        text_length=len(source_text),
    )
    tags = _build_bies_tags(
        num_tokens=len(token_ids),
        predicted_spans=normalized_spans,
        span_class_names=span_class_names,
    )

    label_colors = build_label_color_map(span_class_names)

    text_truncated_chars = 0
    rendered_text = source_text
    if len(rendered_text) > max_chars:
        text_truncated_chars = len(rendered_text) - max_chars
        rendered_text = rendered_text[:max_chars]

    visible_span_views = _clip_span_views(span_views, max_len=len(rendered_text))
    highlighted_text = _render_text_with_spans(
        text=rendered_text,
        span_views=visible_span_views,
        span_class_names=span_class_names,
        label_colors=label_colors,
    )

    lines: list[str] = []
    lines.append(f"preview.example_id: {example_id}")
    lines.append(f"preview.tokens: {len(token_ids)}")
    if mismatch_warning is not None:
        lines.append(mismatch_warning)
    if text_truncated_chars > 0:
        lines.append(f"preview.text_truncated_chars: {text_truncated_chars}")
    lines.append("preview.text:")
    lines.append(highlighted_text if highlighted_text else "(empty)")
    lines.append("preview.predicted_spans:")

    if not span_views:
        lines.append("  (none)")
    else:
        for label_idx, tok_start, tok_end, char_start, char_end in span_views:
            label = _label_name(span_class_names, label_idx)
            snippet = _escape_text_snippet(source_text[char_start:char_end], limit=80)
            label_display = style_labeled_text(
                text=label,
                label=label,
                label_colors=label_colors,
            )
            lines.append(
                f"  {label_display} tok[{tok_start}:{tok_end}] "
                f"char[{char_start}:{char_end}] {snippet}"
            )

    token_count = len(token_ids)
    shown_tokens = min(token_count, max_tokens)
    lines.append(f"preview.token_tags(first {shown_tokens} of {token_count}):")
    lines.append("    idx   token_id   chars        tag                      token")
    for idx in range(shown_tokens):
        tag = tags[idx]
        label_for_tag = _label_from_tag(tag)
        tag_display = style_labeled_text(
            text=tag,
            label=label_for_tag,
            label_colors=label_colors,
        )
        token_display = _escape_text_snippet(token_fragments[idx], limit=48)
        token_display = style_labeled_text(
            text=token_display,
            label=label_for_tag,
            label_colors=label_colors,
        )
        lines.append(
            f"  {idx:5d}  {token_ids[idx]:8d}  {char_starts[idx]:5d}:{char_ends[idx]:<5d}  "
            f"{tag_display:<24} {token_display}"
        )
    if shown_tokens < token_count:
        lines.append(f"  ... {token_count - shown_tokens} more tokens not shown ...")

    return "\n".join(lines)


def _decode_tokens(
    token_ids: Sequence[int], encoding: tiktoken.Encoding
) -> tuple[str, list[str], list[int], list[int]]:
    """Decode token ids into text fragments and cumulative character ranges."""
    fragments: list[str] = []
    char_starts: list[int] = []
    char_ends: list[int] = []
    cursor = 0
    incremental = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for token_id in token_ids:
        char_starts.append(cursor)
        token_bytes = encoding.decode_single_token_bytes(int(token_id))
        fragment = incremental.decode(token_bytes)
        fragments.append(fragment)
        cursor += len(fragment)
        char_ends.append(cursor)
    trailing_fragment = incremental.decode(b"", final=True)
    if trailing_fragment and fragments:
        fragments[-1] = fragments[-1] + trailing_fragment
        cursor += len(trailing_fragment)
        char_ends[-1] = cursor
    return "".join(fragments), fragments, char_starts, char_ends


def _normalize_predicted_spans(
    predicted_spans: Sequence[tuple[int, int, int]],
    token_count: int,
) -> list[tuple[int, int, int]]:
    """Clamp and sort predicted token spans for preview rendering."""
    normalized: list[tuple[int, int, int]] = []
    for label_idx, token_start, token_end in predicted_spans:
        start = int(token_start)
        end = int(token_end)
        if start < 0:
            start = 0
        if end > token_count:
            end = token_count
        if end <= start:
            continue
        normalized.append((int(label_idx), start, end))
    normalized.sort(key=lambda span: (span[1], span[2], span[0]))
    return normalized


def _build_span_views(
    *,
    normalized_spans: Sequence[tuple[int, int, int]],
    char_starts: Sequence[int],
    char_ends: Sequence[int],
    text_length: int,
) -> list[tuple[int, int, int, int, int]]:
    """Convert token spans into preview-ready character span views."""
    views: list[tuple[int, int, int, int, int]] = []
    for label_idx, token_start, token_end in normalized_spans:
        if token_start >= len(char_starts) or token_end <= 0:
            continue
        if token_end - 1 >= len(char_ends):
            continue
        char_start = int(char_starts[token_start])
        char_end = int(char_ends[token_end - 1])
        if char_start < 0:
            char_start = 0
        if char_end > text_length:
            char_end = text_length
        if char_end <= char_start:
            continue
        views.append((label_idx, token_start, token_end, char_start, char_end))
    views.sort(key=lambda view: (view[3], view[4], view[0]))
    return views


def _clip_span_views(
    span_views: Sequence[tuple[int, int, int, int, int]],
    *,
    max_len: int,
) -> list[tuple[int, int, int, int, int]]:
    """Clip preview span views to the visible rendered text length."""
    clipped: list[tuple[int, int, int, int, int]] = []
    for label_idx, token_start, token_end, char_start, char_end in span_views:
        if char_start >= max_len:
            continue
        end = char_end if char_end <= max_len else max_len
        if end <= char_start:
            continue
        clipped.append((label_idx, token_start, token_end, char_start, end))
    return clipped


def _build_bies_tags(
    *,
    num_tokens: int,
    predicted_spans: Sequence[tuple[int, int, int]],
    span_class_names: Sequence[str],
) -> list[str]:
    """Build BIESO tags for the preview token table."""
    tags = [BACKGROUND_CLASS_LABEL] * num_tokens
    for label_idx, token_start, token_end in predicted_spans:
        if not (0 <= token_start < token_end <= num_tokens):
            continue
        label = _label_name(span_class_names, label_idx)
        span_len = token_end - token_start
        if span_len == 1:
            tags[token_start] = f"S-{label}"
            continue
        tags[token_start] = f"B-{label}"
        for idx in range(token_start + 1, token_end - 1):
            tags[idx] = f"I-{label}"
        tags[token_end - 1] = f"E-{label}"
    return tags


def _render_text_with_spans(
    *,
    text: str,
    span_views: Sequence[tuple[int, int, int, int, int]],
    span_class_names: Sequence[str],
    label_colors: dict[str, int],
) -> str:
    """Render text with highlighted spans for the preview block."""
    if not span_views:
        return text
    pieces: list[str] = []
    cursor = 0
    for label_idx, _token_start, _token_end, char_start, char_end in span_views:
        start = max(char_start, cursor)
        end = max(start, char_end)
        if start > len(text):
            break
        if end > len(text):
            end = len(text)
        if end <= start:
            continue
        if start > cursor:
            pieces.append(text[cursor:start])
        segment = text[start:end]
        label = _label_name(span_class_names, label_idx)
        pieces.append(
            style_labeled_text(
                segment,
                label,
                label_colors=label_colors,
            )
        )
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces)


def _label_name(span_class_names: Sequence[str], label_idx: int) -> str:
    """Resolve a display label name for one span label id."""
    if 0 <= label_idx < len(span_class_names):
        return str(span_class_names[label_idx])
    return f"label_{label_idx}"


def _label_from_tag(tag: str) -> str:
    """Extract the span label name from a BIESO tag."""
    if tag == BACKGROUND_CLASS_LABEL:
        return BACKGROUND_CLASS_LABEL
    if "-" not in tag:
        return tag
    return tag.split("-", 1)[1]


def _escape_text_snippet(text: str, *, limit: int) -> str:
    """Escape and truncate one preview text snippet."""
    escaped = text.encode("unicode_escape").decode("ascii")
    if len(escaped) <= limit:
        return escaped
    if limit <= 3:
        return escaped[:limit]
    return escaped[: limit - 3] + "..."
