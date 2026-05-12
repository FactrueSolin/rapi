use std::collections::BTreeMap;

use thiserror::Error;

use super::label_space::LabelInfo;
use super::spans::{ByteSpan, select_non_overlapping_spans};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Typed,
    Redacted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedSpan {
    pub label: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub text: String,
    pub placeholder: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectionSummary {
    pub output_mode: OutputMode,
    pub span_count: usize,
    pub by_label: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactionResult {
    pub text: String,
    pub detected_spans: Vec<DetectedSpan>,
    pub redacted_text: String,
    pub summary: DetectionSummary,
}

#[derive(Error, Debug)]
pub enum RedactionError {
    #[error("unknown span label id {label_id}")]
    UnknownSpanLabel { label_id: usize },

    #[error("invalid redaction span: label={label_id}, bytes={start_byte}..{end_byte}")]
    InvalidSpan {
        label_id: usize,
        start_byte: usize,
        end_byte: usize,
    },

    #[error("redaction spans overlap at byte {start_byte}; spans must be cleaned first")]
    OverlappingSpan { start_byte: usize },
}

impl Default for OutputMode {
    fn default() -> Self {
        Self::Typed
    }
}

pub fn build_redaction_result(
    text: &str,
    spans: &[ByteSpan],
    label_info: &LabelInfo,
    output_mode: OutputMode,
) -> Result<RedactionResult, RedactionError> {
    let non_overlapping = select_non_overlapping_spans(spans);
    let detected_spans = build_detected_spans(text, &non_overlapping, label_info, output_mode)?;
    let redacted_text = redact_text(text, &detected_spans)?;
    let summary = build_detection_summary(output_mode, &detected_spans);

    Ok(RedactionResult {
        text: text.to_string(),
        detected_spans,
        redacted_text,
        summary,
    })
}

pub fn build_detected_spans(
    text: &str,
    spans: &[ByteSpan],
    label_info: &LabelInfo,
    output_mode: OutputMode,
) -> Result<Vec<DetectedSpan>, RedactionError> {
    let mut detected = Vec::with_capacity(spans.len());
    for span in spans {
        validate_span(text, span)?;
        let raw_label = label_info.span_class_names.get(span.label_id).ok_or(
            RedactionError::UnknownSpanLabel {
                label_id: span.label_id,
            },
        )?;
        let (label, placeholder) = match output_mode {
            OutputMode::Typed => (raw_label.clone(), label_placeholder(raw_label)),
            OutputMode::Redacted => ("redacted".to_string(), "<REDACTED>".to_string()),
        };
        detected.push(DetectedSpan {
            label,
            start_byte: span.start_byte,
            end_byte: span.end_byte,
            text: text[span.start_byte..span.end_byte].to_string(),
            placeholder,
        });
    }
    detected.sort_by_key(|span| (span.start_byte, span.end_byte, span.label.clone()));
    Ok(detected)
}

pub fn redact_text(text: &str, spans: &[DetectedSpan]) -> Result<String, RedactionError> {
    if spans.is_empty() {
        return Ok(text.to_string());
    }

    let mut output = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for span in spans {
        if span.start_byte < cursor {
            return Err(RedactionError::OverlappingSpan {
                start_byte: span.start_byte,
            });
        }
        if span.start_byte > span.end_byte
            || span.end_byte > text.len()
            || !text.is_char_boundary(span.start_byte)
            || !text.is_char_boundary(span.end_byte)
        {
            return Err(RedactionError::InvalidSpan {
                label_id: 0,
                start_byte: span.start_byte,
                end_byte: span.end_byte,
            });
        }
        output.push_str(&text[cursor..span.start_byte]);
        output.push_str(&span.placeholder);
        cursor = span.end_byte;
    }
    output.push_str(&text[cursor..]);
    Ok(output)
}

pub fn label_placeholder(label: &str) -> String {
    let mut normalized = String::new();
    let mut last_was_separator = false;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_uppercase());
            last_was_separator = false;
        } else if !last_was_separator && !normalized.is_empty() {
            normalized.push('_');
            last_was_separator = true;
        }
    }
    while normalized.ends_with('_') {
        normalized.pop();
    }
    if normalized.is_empty() {
        normalized = "REDACTED".to_string();
    }
    format!("<{normalized}>")
}

fn build_detection_summary(output_mode: OutputMode, spans: &[DetectedSpan]) -> DetectionSummary {
    let mut by_label = BTreeMap::new();
    for span in spans {
        *by_label.entry(span.label.clone()).or_insert(0) += 1;
    }
    DetectionSummary {
        output_mode,
        span_count: spans.len(),
        by_label,
    }
}

fn validate_span(text: &str, span: &ByteSpan) -> Result<(), RedactionError> {
    if span.start_byte >= span.end_byte
        || span.end_byte > text.len()
        || !text.is_char_boundary(span.start_byte)
        || !text.is_char_boundary(span.end_byte)
    {
        return Err(RedactionError::InvalidSpan {
            label_id: span.label_id,
            start_byte: span.start_byte,
            end_byte: span.end_byte,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label_info() -> LabelInfo {
        LabelInfo {
            category_version: "test".to_string(),
            token_class_names: vec!["O".to_string(), "S-email".to_string()],
            span_class_names: vec!["O".to_string(), "personal_email".to_string()],
            token_to_span_label: vec![0, 1],
            token_boundary_tags: vec![None, Some(super::super::label_space::BoundaryTag::Single)],
            background_token_label: 0,
            background_span_label: 0,
        }
    }

    #[test]
    fn builds_typed_redaction_result() {
        let text = "email john@example.com now";
        let result = build_redaction_result(
            text,
            &[ByteSpan {
                label_id: 1,
                start_byte: 6,
                end_byte: 22,
            }],
            &label_info(),
            OutputMode::Typed,
        )
        .unwrap();

        assert_eq!(result.redacted_text, "email <PERSONAL_EMAIL> now");
        assert_eq!(result.detected_spans[0].text, "john@example.com");
        assert_eq!(result.summary.by_label["personal_email"], 1);
    }

    #[test]
    fn builds_redacted_output_mode() {
        let text = "email john@example.com";
        let result = build_redaction_result(
            text,
            &[ByteSpan {
                label_id: 1,
                start_byte: 6,
                end_byte: 22,
            }],
            &label_info(),
            OutputMode::Redacted,
        )
        .unwrap();

        assert_eq!(result.redacted_text, "email <REDACTED>");
        assert_eq!(result.detected_spans[0].label, "redacted");
    }

    #[test]
    fn placeholder_normalizes_non_alphanumeric_runs() {
        assert_eq!(label_placeholder("secret-url"), "<SECRET_URL>");
        assert_eq!(label_placeholder("---"), "<REDACTED>");
    }

    #[test]
    fn redacts_unicode_byte_spans() {
        let text = "邮箱是 test@example.com";
        let result = build_redaction_result(
            text,
            &[ByteSpan {
                label_id: 1,
                start_byte: 10,
                end_byte: 26,
            }],
            &label_info(),
            OutputMode::Typed,
        )
        .unwrap();

        assert_eq!(result.redacted_text, "邮箱是 <PERSONAL_EMAIL>");
    }
}
