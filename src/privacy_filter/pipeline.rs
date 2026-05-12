use std::time::Instant;

use thiserror::Error;

use super::decoding::{DecodeError, DecodeMode, DecoderOptions, SequenceDecoder};
use super::label_space::{LabelInfo, LabelSpaceError};
use super::model_manager::ResolvedModelPaths;
use super::onnx_session::{OnnxSessionError, OnnxSessionOptions, PrivacyFilterOnnxSession};
use super::redaction::{OutputMode, RedactionError, RedactionResult, build_redaction_result};
use super::scoring::{ScoringError, aggregate_window_logits};
use super::spans::{
    SpanError, discard_overlapping_spans_by_label, labels_to_token_spans,
    token_spans_to_byte_spans, trim_byte_spans_whitespace,
};
use super::tokenizer::{PrivacyFilterTokenizer, TokenizerError, TokenizerOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivacyFilterPipelineOptions {
    pub decode_mode: DecodeMode,
    pub output_mode: OutputMode,
    pub trim_span_whitespace: bool,
    pub discard_overlapping_predicted_spans: bool,
    pub use_viterbi_calibration: bool,
}

#[derive(Debug)]
pub struct PrivacyFilterPipeline {
    tokenizer: PrivacyFilterTokenizer,
    session: PrivacyFilterOnnxSession,
    label_info: LabelInfo,
    decoder: SequenceDecoder,
    options: PrivacyFilterPipelineOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrivacyFilterPipelineMetrics {
    pub text_bytes: usize,
    pub token_count: usize,
    pub window_count: usize,
    pub detected_span_count: usize,
    pub tokenize_ms: f64,
    pub window_ms: f64,
    pub onnx_ms: f64,
    pub scoring_ms: f64,
    pub decode_ms: f64,
    pub span_ms: f64,
    pub redaction_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrivacyFilterPipelineRun {
    pub result: RedactionResult,
    pub metrics: PrivacyFilterPipelineMetrics,
}

#[derive(Error, Debug)]
pub enum PrivacyFilterPipelineError {
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),

    #[error(transparent)]
    Onnx(#[from] OnnxSessionError),

    #[error(transparent)]
    LabelSpace(#[from] LabelSpaceError),

    #[error(transparent)]
    Scoring(#[from] ScoringError),

    #[error(transparent)]
    Decode(#[from] DecodeError),

    #[error(transparent)]
    Span(#[from] SpanError),

    #[error(transparent)]
    Redaction(#[from] RedactionError),
}

impl Default for PrivacyFilterPipelineOptions {
    fn default() -> Self {
        Self {
            decode_mode: DecodeMode::Viterbi,
            output_mode: OutputMode::Typed,
            trim_span_whitespace: true,
            discard_overlapping_predicted_spans: false,
            use_viterbi_calibration: true,
        }
    }
}

impl PrivacyFilterPipeline {
    pub fn from_paths(
        paths: &ResolvedModelPaths,
        tokenizer_options: TokenizerOptions,
        session_options: OnnxSessionOptions,
        pipeline_options: PrivacyFilterPipelineOptions,
    ) -> Result<Self, PrivacyFilterPipelineError> {
        let tokenizer = PrivacyFilterTokenizer::from_paths(paths, tokenizer_options)?;
        let session = PrivacyFilterOnnxSession::from_paths(paths, session_options)?;
        let label_info = LabelInfo::from_paths(paths)?;
        let decoder = SequenceDecoder::from_paths(
            paths,
            label_info.clone(),
            DecoderOptions {
                mode: pipeline_options.decode_mode,
                use_viterbi_calibration: pipeline_options.use_viterbi_calibration,
                viterbi_calibration_path: None,
            },
        )?;

        Ok(Self {
            tokenizer,
            session,
            label_info,
            decoder,
            options: pipeline_options,
        })
    }

    pub fn tokenizer(&self) -> &PrivacyFilterTokenizer {
        &self.tokenizer
    }

    pub fn session(&self) -> &PrivacyFilterOnnxSession {
        &self.session
    }

    pub fn label_info(&self) -> &LabelInfo {
        &self.label_info
    }

    pub fn options(&self) -> &PrivacyFilterPipelineOptions {
        &self.options
    }

    pub fn redact(&mut self, text: &str) -> Result<RedactionResult, PrivacyFilterPipelineError> {
        Ok(self.redact_with_metrics(text)?.result)
    }

    pub fn redact_with_metrics(
        &mut self,
        text: &str,
    ) -> Result<PrivacyFilterPipelineRun, PrivacyFilterPipelineError> {
        let total_start = Instant::now();

        let stage_start = Instant::now();
        let tokenized = self.tokenizer.encode(text)?;
        let tokenize_ms = elapsed_ms(stage_start);

        if tokenized.token_ids.is_empty() {
            let stage_start = Instant::now();
            let result =
                build_redaction_result(text, &[], &self.label_info, self.options.output_mode)?;
            let redaction_ms = elapsed_ms(stage_start);
            return Ok(PrivacyFilterPipelineRun {
                metrics: PrivacyFilterPipelineMetrics {
                    text_bytes: text.len(),
                    token_count: 0,
                    window_count: 0,
                    detected_span_count: result.detected_spans.len(),
                    tokenize_ms,
                    window_ms: 0.0,
                    onnx_ms: 0.0,
                    scoring_ms: 0.0,
                    decode_ms: 0.0,
                    span_ms: 0.0,
                    redaction_ms,
                    total_ms: elapsed_ms(total_start),
                },
                result,
            });
        }

        let stage_start = Instant::now();
        let windows = self.tokenizer.windows(&tokenized)?;
        let window_ms = elapsed_ms(stage_start);

        let stage_start = Instant::now();
        let window_logits = self.session.run_windows(&windows)?;
        let onnx_ms = elapsed_ms(stage_start);

        let stage_start = Instant::now();
        let token_scores = aggregate_window_logits(
            &window_logits,
            tokenized.token_ids.len(),
            self.label_info.label_count(),
        )?;
        let scoring_ms = elapsed_ms(stage_start);

        let stage_start = Instant::now();
        let labels_by_token = self.decoder.decode(&token_scores)?;
        let decode_ms = elapsed_ms(stage_start);

        let stage_start = Instant::now();
        let token_spans = labels_to_token_spans(&labels_by_token, &self.label_info)?;
        let mut byte_spans = token_spans_to_byte_spans(
            &token_spans,
            &tokenized.token_offsets,
            &tokenized.original_text,
        )?;

        if self.options.trim_span_whitespace {
            byte_spans = trim_byte_spans_whitespace(&byte_spans, &tokenized.original_text)?;
        }
        if self.options.discard_overlapping_predicted_spans {
            byte_spans = discard_overlapping_spans_by_label(&byte_spans);
        }
        let span_ms = elapsed_ms(stage_start);

        let stage_start = Instant::now();
        let result = build_redaction_result(
            &tokenized.original_text,
            &byte_spans,
            &self.label_info,
            self.options.output_mode,
        )?;
        let redaction_ms = elapsed_ms(stage_start);

        Ok(PrivacyFilterPipelineRun {
            metrics: PrivacyFilterPipelineMetrics {
                text_bytes: text.len(),
                token_count: tokenized.token_ids.len(),
                window_count: windows.len(),
                detected_span_count: result.detected_spans.len(),
                tokenize_ms,
                window_ms,
                onnx_ms,
                scoring_ms,
                decode_ms,
                span_ms,
                redaction_ms,
                total_ms: elapsed_ms(total_start),
            },
            result,
        })
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_match_privacy_filter_pipeline_defaults() {
        let options = PrivacyFilterPipelineOptions::default();

        assert_eq!(options.decode_mode, DecodeMode::Viterbi);
        assert_eq!(options.output_mode, OutputMode::Typed);
        assert!(options.trim_span_whitespace);
        assert!(!options.discard_overlapping_predicted_spans);
        assert!(options.use_viterbi_calibration);
    }
}
