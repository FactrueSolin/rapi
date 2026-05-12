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
        let tokenized = self.tokenizer.encode(text)?;
        if tokenized.token_ids.is_empty() {
            return Ok(build_redaction_result(
                text,
                &[],
                &self.label_info,
                self.options.output_mode,
            )?);
        }

        let windows = self.tokenizer.windows(&tokenized)?;
        let window_logits = self.session.run_windows(&windows)?;
        let token_scores = aggregate_window_logits(
            &window_logits,
            tokenized.token_ids.len(),
            self.label_info.label_count(),
        )?;
        let labels_by_token = self.decoder.decode(&token_scores)?;
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

        Ok(build_redaction_result(
            &tokenized.original_text,
            &byte_spans,
            &self.label_info,
            self.options.output_mode,
        )?)
    }
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
