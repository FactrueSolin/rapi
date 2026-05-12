use std::path::PathBuf;

use serde::Deserialize;
use thiserror::Error;

use super::label_space::{BoundaryTag, LabelInfo};
use super::model_manager::ResolvedModelPaths;
use super::scoring::TokenScores;

const NEG_INF: f32 = -1e9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    Viterbi,
    Argmax,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderOptions {
    pub mode: DecodeMode,
    pub use_viterbi_calibration: bool,
    pub viterbi_calibration_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViterbiTransitionBiases {
    pub transition_bias_background_stay: f32,
    pub transition_bias_background_to_start: f32,
    pub transition_bias_inside_to_continue: f32,
    pub transition_bias_inside_to_end: f32,
    pub transition_bias_end_to_background: f32,
    pub transition_bias_end_to_start: f32,
}

#[derive(Debug, Clone)]
pub struct SequenceDecoder {
    mode: DecodeMode,
    label_info: LabelInfo,
    biases: ViterbiTransitionBiases,
    start_scores: Vec<f32>,
    end_scores: Vec<f32>,
    transition_scores: Vec<Vec<f32>>,
}

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("viterbi calibration read failed for {path}: {source}")]
    CalibrationRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("viterbi calibration parse failed for {path}: {source}")]
    CalibrationParse {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("viterbi calibration is invalid: {0}")]
    InvalidCalibration(String),

    #[error("score row {row_index} has label count {actual}, expected {expected}")]
    LabelCountMismatch {
        row_index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("score row {row_index} contains non-finite label {label_index}: {value}")]
    NonFiniteScore {
        row_index: usize,
        label_index: usize,
        value: f32,
    },
}

impl Default for DecoderOptions {
    fn default() -> Self {
        Self {
            mode: DecodeMode::Viterbi,
            use_viterbi_calibration: true,
            viterbi_calibration_path: None,
        }
    }
}

impl Default for ViterbiTransitionBiases {
    fn default() -> Self {
        Self {
            transition_bias_background_stay: 0.0,
            transition_bias_background_to_start: 0.0,
            transition_bias_inside_to_continue: 0.0,
            transition_bias_inside_to_end: 0.0,
            transition_bias_end_to_background: 0.0,
            transition_bias_end_to_start: 0.0,
        }
    }
}

impl SequenceDecoder {
    pub fn from_paths(
        paths: &ResolvedModelPaths,
        label_info: LabelInfo,
        options: DecoderOptions,
    ) -> Result<Self, DecodeError> {
        let calibration_path = options
            .viterbi_calibration_path
            .clone()
            .unwrap_or_else(|| paths.viterbi_calibration_path.clone());
        let biases = if options.mode == DecodeMode::Viterbi
            && options.use_viterbi_calibration
            && calibration_path.exists()
        {
            read_viterbi_transition_biases(&calibration_path)?
        } else {
            ViterbiTransitionBiases::default()
        };
        Self::new(label_info, options.mode, biases)
    }

    pub fn new(
        label_info: LabelInfo,
        mode: DecodeMode,
        biases: ViterbiTransitionBiases,
    ) -> Result<Self, DecodeError> {
        let (start_scores, end_scores, transition_scores) =
            build_viterbi_scores(&label_info, &biases);
        Ok(Self {
            mode,
            label_info,
            biases,
            start_scores,
            end_scores,
            transition_scores,
        })
    }

    pub fn mode(&self) -> DecodeMode {
        self.mode
    }

    pub fn biases(&self) -> &ViterbiTransitionBiases {
        &self.biases
    }

    pub fn decode(&self, token_scores: &TokenScores) -> Result<Vec<(usize, usize)>, DecodeError> {
        validate_score_rows(&token_scores.logprobs, self.label_info.label_count())?;
        let labels = match self.mode {
            DecodeMode::Argmax => decode_argmax(&token_scores.logprobs),
            DecodeMode::Viterbi => self.decode_viterbi(&token_scores.logprobs),
        };
        Ok(token_scores
            .token_positions
            .iter()
            .copied()
            .zip(labels)
            .collect())
    }

    fn decode_viterbi(&self, logprobs: &[Vec<f32>]) -> Vec<usize> {
        if logprobs.is_empty() {
            return Vec::new();
        }

        let seq_len = logprobs.len();
        let label_count = self.label_info.label_count();
        let mut scores = vec![NEG_INF; label_count];
        for (label_id, score) in scores.iter_mut().enumerate() {
            *score = logprobs[0][label_id] + self.start_scores[label_id];
        }

        let mut backpointers = vec![vec![0usize; label_count]; seq_len.saturating_sub(1)];
        for step in 1..seq_len {
            let mut next_scores = vec![NEG_INF; label_count];
            for next_label in 0..label_count {
                let mut best_score = NEG_INF;
                let mut best_prev = 0usize;
                for (prev_label, prev_score) in scores.iter().enumerate() {
                    let candidate = *prev_score + self.transition_scores[prev_label][next_label];
                    if candidate > best_score {
                        best_score = candidate;
                        best_prev = prev_label;
                    }
                }
                next_scores[next_label] = best_score + logprobs[step][next_label];
                backpointers[step - 1][next_label] = best_prev;
            }
            scores = next_scores;
        }

        if !scores
            .iter()
            .any(|score| score.is_finite() && *score > NEG_INF / 2.0)
        {
            return decode_argmax(logprobs);
        }

        for (label_id, score) in scores.iter_mut().enumerate() {
            *score += self.end_scores[label_id];
        }
        let Some((mut last_label, _)) = scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
        else {
            return Vec::new();
        };

        let mut path = vec![0usize; seq_len];
        path[seq_len - 1] = last_label;
        for step in (0..seq_len - 1).rev() {
            last_label = backpointers[step][last_label];
            path[step] = last_label;
        }
        path
    }
}

pub fn decode_argmax(logprobs: &[Vec<f32>]) -> Vec<usize> {
    logprobs
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(label_id, _)| label_id)
                .unwrap_or(0)
        })
        .collect()
}

fn validate_score_rows(logprobs: &[Vec<f32>], label_count: usize) -> Result<(), DecodeError> {
    for (row_index, row) in logprobs.iter().enumerate() {
        if row.len() != label_count {
            return Err(DecodeError::LabelCountMismatch {
                row_index,
                expected: label_count,
                actual: row.len(),
            });
        }
        for (label_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(DecodeError::NonFiniteScore {
                    row_index,
                    label_index,
                    value: *value,
                });
            }
        }
    }
    Ok(())
}

fn build_viterbi_scores(
    label_info: &LabelInfo,
    biases: &ViterbiTransitionBiases,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let label_count = label_info.label_count();
    let mut start_scores = vec![NEG_INF; label_count];
    let mut end_scores = vec![NEG_INF; label_count];
    let mut transition_scores = vec![vec![NEG_INF; label_count]; label_count];

    for label_id in 0..label_count {
        let tag = label_info.token_boundary_tags[label_id];
        if label_id == label_info.background_token_label
            || matches!(tag, Some(BoundaryTag::Begin | BoundaryTag::Single))
        {
            start_scores[label_id] = 0.0;
        }
        if label_id == label_info.background_token_label
            || matches!(tag, Some(BoundaryTag::End | BoundaryTag::Single))
        {
            end_scores[label_id] = 0.0;
        }

        for next_label in 0..label_count {
            if is_valid_transition(label_info, label_id, next_label) {
                transition_scores[label_id][next_label] =
                    transition_bias(label_info, biases, label_id, next_label);
            }
        }
    }

    (start_scores, end_scores, transition_scores)
}

fn is_valid_transition(label_info: &LabelInfo, prev_label: usize, next_label: usize) -> bool {
    let prev_span = label_info.token_to_span_label[prev_label];
    let next_span = label_info.token_to_span_label[next_label];
    let prev_tag = label_info.token_boundary_tags[prev_label];
    let next_tag = label_info.token_boundary_tags[next_label];
    let prev_is_background = prev_span == label_info.background_span_label
        || prev_label == label_info.background_token_label;
    let next_is_background = next_span == label_info.background_span_label
        || next_label == label_info.background_token_label;

    if prev_is_background {
        return next_is_background
            || matches!(next_tag, Some(BoundaryTag::Begin | BoundaryTag::Single));
    }
    if matches!(prev_tag, Some(BoundaryTag::Begin | BoundaryTag::Inside)) {
        return prev_span == next_span
            && matches!(next_tag, Some(BoundaryTag::Inside | BoundaryTag::End));
    }
    if matches!(prev_tag, Some(BoundaryTag::End | BoundaryTag::Single)) {
        return next_is_background
            || matches!(next_tag, Some(BoundaryTag::Begin | BoundaryTag::Single));
    }
    false
}

fn transition_bias(
    label_info: &LabelInfo,
    biases: &ViterbiTransitionBiases,
    prev_label: usize,
    next_label: usize,
) -> f32 {
    let prev_span = label_info.token_to_span_label[prev_label];
    let next_span = label_info.token_to_span_label[next_label];
    let prev_tag = label_info.token_boundary_tags[prev_label];
    let next_tag = label_info.token_boundary_tags[next_label];
    let prev_is_background = prev_span == label_info.background_span_label
        || prev_label == label_info.background_token_label;
    let next_is_background = next_span == label_info.background_span_label
        || next_label == label_info.background_token_label;

    if prev_is_background {
        if next_is_background {
            return biases.transition_bias_background_stay;
        }
        if matches!(next_tag, Some(BoundaryTag::Begin | BoundaryTag::Single)) {
            return biases.transition_bias_background_to_start;
        }
        return 0.0;
    }

    if matches!(prev_tag, Some(BoundaryTag::Begin | BoundaryTag::Inside)) {
        if prev_span == next_span && next_tag == Some(BoundaryTag::Inside) {
            return biases.transition_bias_inside_to_continue;
        }
        if prev_span == next_span && next_tag == Some(BoundaryTag::End) {
            return biases.transition_bias_inside_to_end;
        }
        return 0.0;
    }

    if matches!(prev_tag, Some(BoundaryTag::End | BoundaryTag::Single)) {
        if next_is_background {
            return biases.transition_bias_end_to_background;
        }
        if matches!(next_tag, Some(BoundaryTag::Begin | BoundaryTag::Single)) {
            return biases.transition_bias_end_to_start;
        }
    }

    0.0
}

#[derive(Debug, Deserialize)]
struct CalibrationArtifact {
    operating_points: CalibrationOperatingPoints,
}

#[derive(Debug, Deserialize)]
struct CalibrationOperatingPoints {
    default: CalibrationDefaultEntry,
}

#[derive(Debug, Deserialize)]
struct CalibrationDefaultEntry {
    biases: ViterbiTransitionBiasesFile,
}

#[derive(Debug, Deserialize)]
struct ViterbiTransitionBiasesFile {
    transition_bias_background_stay: f32,
    transition_bias_background_to_start: f32,
    transition_bias_inside_to_continue: f32,
    transition_bias_inside_to_end: f32,
    transition_bias_end_to_background: f32,
    transition_bias_end_to_start: f32,
}

fn read_viterbi_transition_biases(path: &PathBuf) -> Result<ViterbiTransitionBiases, DecodeError> {
    let content = std::fs::read_to_string(path).map_err(|source| DecodeError::CalibrationRead {
        path: path.clone(),
        source,
    })?;
    let artifact: CalibrationArtifact =
        serde_json::from_str(&content).map_err(|source| DecodeError::CalibrationParse {
            path: path.clone(),
            source,
        })?;
    let biases = artifact.operating_points.default.biases;
    let resolved = ViterbiTransitionBiases {
        transition_bias_background_stay: biases.transition_bias_background_stay,
        transition_bias_background_to_start: biases.transition_bias_background_to_start,
        transition_bias_inside_to_continue: biases.transition_bias_inside_to_continue,
        transition_bias_inside_to_end: biases.transition_bias_inside_to_end,
        transition_bias_end_to_background: biases.transition_bias_end_to_background,
        transition_bias_end_to_start: biases.transition_bias_end_to_start,
    };
    for value in [
        resolved.transition_bias_background_stay,
        resolved.transition_bias_background_to_start,
        resolved.transition_bias_inside_to_continue,
        resolved.transition_bias_inside_to_end,
        resolved.transition_bias_end_to_background,
        resolved.transition_bias_end_to_start,
    ] {
        if !value.is_finite() {
            return Err(DecodeError::InvalidCalibration(
                "transition biases must be finite".to_string(),
            ));
        }
    }
    Ok(resolved)
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
                "B-phone".to_string(),
                "I-phone".to_string(),
                "E-phone".to_string(),
                "S-phone".to_string(),
            ],
            span_class_names: vec!["O".to_string(), "email".to_string(), "phone".to_string()],
            token_to_span_label: vec![0, 1, 1, 1, 1, 2, 2, 2, 2],
            token_boundary_tags: vec![
                None,
                Some(BoundaryTag::Begin),
                Some(BoundaryTag::Inside),
                Some(BoundaryTag::End),
                Some(BoundaryTag::Single),
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
    fn argmax_selects_best_label() {
        let labels = decode_argmax(&[vec![-2.0, -1.0], vec![0.0, -3.0]]);

        assert_eq!(labels, vec![1, 0]);
    }

    #[test]
    fn viterbi_avoids_cross_label_inside_transition() {
        let decoder = SequenceDecoder::new(
            label_info(),
            DecodeMode::Viterbi,
            ViterbiTransitionBiases::default(),
        )
        .unwrap();
        let scores = TokenScores {
            token_positions: vec![0, 1],
            logprobs: vec![
                vec![-10.0, 0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                vec![-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, -1.0, -10.0],
            ],
        };

        let decoded = decoder.decode(&scores).unwrap();

        assert_ne!(decoded[1].1, 6);
        assert_eq!(decoded[1].1, 3);
    }

    #[test]
    fn transition_bias_can_prefer_starting_entity() {
        let mut biases = ViterbiTransitionBiases::default();
        biases.transition_bias_background_to_start = 5.0;
        let decoder = SequenceDecoder::new(label_info(), DecodeMode::Viterbi, biases).unwrap();
        let scores = TokenScores {
            token_positions: vec![0, 1],
            logprobs: vec![
                vec![0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                vec![0.0, -1.0, -10.0, -10.0, -1.0, -10.0, -10.0, -10.0, -10.0],
            ],
        };

        let decoded = decoder.decode(&scores).unwrap();

        assert_eq!(decoded[1].1, 4);
    }
}
