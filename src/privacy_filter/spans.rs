use std::collections::HashMap;

use thiserror::Error;

use super::label_space::{BoundaryTag, LabelInfo};
use super::tokenizer::TokenOffset;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    pub label_id: usize,
    pub start_token: usize,
    pub end_token: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSpan {
    pub label_id: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Error, Debug)]
pub enum SpanError {
    #[error("unknown token label id {label_id} at token index {token_index}")]
    UnknownTokenLabel { token_index: usize, label_id: usize },

    #[error("invalid token span: label={label_id}, tokens={start_token}..{end_token}")]
    InvalidTokenSpan {
        label_id: usize,
        start_token: usize,
        end_token: usize,
    },

    #[error("missing token offset for token index {token_index}")]
    MissingTokenOffset { token_index: usize },

    #[error("invalid byte span: label={label_id}, bytes={start_byte}..{end_byte}")]
    InvalidByteSpan {
        label_id: usize,
        start_byte: usize,
        end_byte: usize,
    },
}

pub fn labels_to_token_spans(
    labels_by_token_index: &[(usize, usize)],
    label_info: &LabelInfo,
) -> Result<Vec<TokenSpan>, SpanError> {
    let mut ordered = labels_by_token_index.to_vec();
    ordered.sort_by_key(|(token_index, _)| *token_index);

    let mut spans = Vec::new();
    let mut current_label = None;
    let mut start_token = None;
    let mut previous_token = None;

    for (token_index, label_id) in ordered {
        if label_id >= label_info.token_to_span_label.len() {
            return Err(SpanError::UnknownTokenLabel {
                token_index,
                label_id,
            });
        }

        if let Some(previous) = previous_token {
            if token_index != previous + 1 {
                close_current_span(
                    &mut spans,
                    &mut current_label,
                    &mut start_token,
                    previous + 1,
                );
            }
        }

        let span_label = label_info.token_to_span_label[label_id];
        let boundary_tag = label_info.token_boundary_tags[label_id];
        let is_background = span_label == label_info.background_span_label
            || label_id == label_info.background_token_label;

        if is_background {
            close_current_span(
                &mut spans,
                &mut current_label,
                &mut start_token,
                token_index,
            );
            previous_token = Some(token_index);
            continue;
        }

        match boundary_tag {
            Some(BoundaryTag::Single) => {
                if let Some(previous) = previous_token {
                    close_current_span(
                        &mut spans,
                        &mut current_label,
                        &mut start_token,
                        previous + 1,
                    );
                }
                spans.push(TokenSpan {
                    label_id: span_label,
                    start_token: token_index,
                    end_token: token_index + 1,
                });
                current_label = None;
                start_token = None;
            }
            Some(BoundaryTag::Begin) => {
                if let Some(previous) = previous_token {
                    close_current_span(
                        &mut spans,
                        &mut current_label,
                        &mut start_token,
                        previous + 1,
                    );
                }
                current_label = Some(span_label);
                start_token = Some(token_index);
            }
            Some(BoundaryTag::Inside) => {
                if current_label != Some(span_label) {
                    if let Some(previous) = previous_token {
                        close_current_span(
                            &mut spans,
                            &mut current_label,
                            &mut start_token,
                            previous + 1,
                        );
                    }
                    current_label = Some(span_label);
                    start_token = Some(token_index);
                }
            }
            Some(BoundaryTag::End) => {
                if current_label == Some(span_label) {
                    let start = start_token.unwrap_or(token_index);
                    spans.push(TokenSpan {
                        label_id: span_label,
                        start_token: start,
                        end_token: token_index + 1,
                    });
                    current_label = None;
                    start_token = None;
                } else {
                    if let Some(previous) = previous_token {
                        close_current_span(
                            &mut spans,
                            &mut current_label,
                            &mut start_token,
                            previous + 1,
                        );
                    }
                    spans.push(TokenSpan {
                        label_id: span_label,
                        start_token: token_index,
                        end_token: token_index + 1,
                    });
                }
            }
            None => {
                close_current_span(
                    &mut spans,
                    &mut current_label,
                    &mut start_token,
                    token_index,
                );
            }
        }

        previous_token = Some(token_index);
    }

    if let Some(previous) = previous_token {
        close_current_span(
            &mut spans,
            &mut current_label,
            &mut start_token,
            previous + 1,
        );
    }

    Ok(spans)
}

pub fn token_spans_to_byte_spans(
    token_spans: &[TokenSpan],
    token_offsets: &[TokenOffset],
    text: &str,
) -> Result<Vec<ByteSpan>, SpanError> {
    let offsets_by_index = token_offsets
        .iter()
        .map(|offset| (offset.token_index, offset))
        .collect::<HashMap<_, _>>();
    let mut byte_spans = Vec::new();

    for span in token_spans {
        if span.start_token >= span.end_token {
            return Err(SpanError::InvalidTokenSpan {
                label_id: span.label_id,
                start_token: span.start_token,
                end_token: span.end_token,
            });
        }
        let start_offset =
            offsets_by_index
                .get(&span.start_token)
                .ok_or(SpanError::MissingTokenOffset {
                    token_index: span.start_token,
                })?;
        let end_offset =
            offsets_by_index
                .get(&(span.end_token - 1))
                .ok_or(SpanError::MissingTokenOffset {
                    token_index: span.end_token - 1,
                })?;
        let byte_span = ByteSpan {
            label_id: span.label_id,
            start_byte: start_offset.start_byte,
            end_byte: end_offset.end_byte,
        };
        validate_byte_span(&byte_span, text)?;
        if byte_span.end_byte > byte_span.start_byte {
            byte_spans.push(byte_span);
        }
    }

    Ok(byte_spans)
}

pub fn trim_byte_spans_whitespace(
    spans: &[ByteSpan],
    text: &str,
) -> Result<Vec<ByteSpan>, SpanError> {
    let mut trimmed = Vec::new();
    for span in spans {
        validate_byte_span(span, text)?;
        let slice = &text[span.start_byte..span.end_byte];
        let leading = slice
            .char_indices()
            .find(|(_, ch)| !ch.is_whitespace())
            .map(|(index, _)| index);
        let Some(leading) = leading else {
            continue;
        };
        let trailing = slice
            .char_indices()
            .rev()
            .find(|(_, ch)| !ch.is_whitespace())
            .map(|(index, ch)| index + ch.len_utf8())
            .unwrap_or(leading);
        let next = ByteSpan {
            label_id: span.label_id,
            start_byte: span.start_byte + leading,
            end_byte: span.start_byte + trailing,
        };
        if next.end_byte > next.start_byte {
            trimmed.push(next);
        }
    }
    Ok(trimmed)
}

pub fn discard_overlapping_spans_by_label(spans: &[ByteSpan]) -> Vec<ByteSpan> {
    let mut by_label: HashMap<usize, Vec<ByteSpan>> = HashMap::new();
    for span in spans {
        by_label
            .entry(span.label_id)
            .or_default()
            .push(span.clone());
    }

    let mut kept = Vec::new();
    for spans in by_label.values_mut() {
        spans.sort_by_key(|span| {
            (
                span.start_byte,
                usize::MAX - (span.end_byte - span.start_byte),
            )
        });
        let mut kept_for_label: Vec<ByteSpan> = Vec::new();
        'candidate: for span in spans.iter() {
            for existing in &kept_for_label {
                if spans_overlap(span, existing) {
                    continue 'candidate;
                }
            }
            kept_for_label.push(span.clone());
        }
        kept.extend(kept_for_label);
    }
    kept.sort_by_key(|span| (span.start_byte, span.end_byte, span.label_id));
    kept
}

pub fn select_non_overlapping_spans(spans: &[ByteSpan]) -> Vec<ByteSpan> {
    let mut ordered = spans.to_vec();
    ordered.sort_by_key(|span| {
        (
            span.start_byte,
            usize::MAX - (span.end_byte - span.start_byte),
            span.label_id,
        )
    });

    let mut kept = Vec::new();
    let mut cursor = 0usize;
    for span in ordered {
        if span.start_byte < cursor || span.end_byte <= span.start_byte {
            continue;
        }
        cursor = span.end_byte;
        kept.push(span);
    }
    kept
}

fn close_current_span(
    spans: &mut Vec<TokenSpan>,
    current_label: &mut Option<usize>,
    start_token: &mut Option<usize>,
    end_token: usize,
) {
    if let (Some(label_id), Some(start_token)) = (*current_label, *start_token) {
        if end_token > start_token {
            spans.push(TokenSpan {
                label_id,
                start_token,
                end_token,
            });
        }
    }
    *current_label = None;
    *start_token = None;
}

fn validate_byte_span(span: &ByteSpan, text: &str) -> Result<(), SpanError> {
    if span.start_byte > span.end_byte
        || span.end_byte > text.len()
        || !text.is_char_boundary(span.start_byte)
        || !text.is_char_boundary(span.end_byte)
    {
        return Err(SpanError::InvalidByteSpan {
            label_id: span.label_id,
            start_byte: span.start_byte,
            end_byte: span.end_byte,
        });
    }
    Ok(())
}

fn spans_overlap(left: &ByteSpan, right: &ByteSpan) -> bool {
    left.start_byte < right.end_byte && right.start_byte < left.end_byte
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label_info() -> LabelInfo {
        LabelInfo {
            category_version: "test".to_string(),
            token_class_names: vec![
                "O".to_string(),
                "B-email".to_string(),
                "I-email".to_string(),
                "E-email".to_string(),
                "S-email".to_string(),
            ],
            span_class_names: vec!["O".to_string(), "email".to_string()],
            token_to_span_label: vec![0, 1, 1, 1, 1],
            token_boundary_tags: vec![
                None,
                Some(BoundaryTag::Begin),
                Some(BoundaryTag::Inside),
                Some(BoundaryTag::End),
                Some(BoundaryTag::Single),
            ],
            background_token_label: 0,
            background_span_label: 0,
        }
    }

    #[test]
    fn converts_bioes_labels_to_token_spans() {
        let spans = labels_to_token_spans(&[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], &label_info())
            .unwrap();

        assert_eq!(
            spans,
            vec![
                TokenSpan {
                    label_id: 1,
                    start_token: 1,
                    end_token: 4
                },
                TokenSpan {
                    label_id: 1,
                    start_token: 4,
                    end_token: 5
                }
            ]
        );
    }

    #[test]
    fn handles_illegal_inside_as_new_span() {
        let spans = labels_to_token_spans(&[(0, 2), (1, 3)], &label_info()).unwrap();

        assert_eq!(
            spans,
            vec![TokenSpan {
                label_id: 1,
                start_token: 0,
                end_token: 2
            }]
        );
    }

    #[test]
    fn converts_token_span_to_byte_span() {
        let text = "hi 邮箱";
        let offsets = vec![
            TokenOffset {
                token_index: 0,
                start_byte: 0,
                end_byte: 2,
            },
            TokenOffset {
                token_index: 1,
                start_byte: 3,
                end_byte: 9,
            },
        ];

        let spans = token_spans_to_byte_spans(
            &[TokenSpan {
                label_id: 1,
                start_token: 1,
                end_token: 2,
            }],
            &offsets,
            text,
        )
        .unwrap();

        assert_eq!(spans[0].start_byte, 3);
        assert_eq!(spans[0].end_byte, 9);
    }

    #[test]
    fn trims_unicode_whitespace_safely() {
        let text = "x  邮箱  y";
        let spans = trim_byte_spans_whitespace(
            &[ByteSpan {
                label_id: 1,
                start_byte: 1,
                end_byte: 11,
            }],
            text,
        )
        .unwrap();

        assert_eq!(
            spans[0],
            ByteSpan {
                label_id: 1,
                start_byte: 3,
                end_byte: 9
            }
        );
    }
}
