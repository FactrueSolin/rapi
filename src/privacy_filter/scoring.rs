use thiserror::Error;

use super::onnx_session::WindowLogits;

#[derive(Debug, Clone, PartialEq)]
pub struct TokenScores {
    pub token_positions: Vec<usize>,
    pub logprobs: Vec<Vec<f32>>,
}

#[derive(Error, Debug)]
pub enum ScoringError {
    #[error("window logits length mismatch: token_indices={token_indices}, logits={logits}")]
    WindowLengthMismatch { token_indices: usize, logits: usize },

    #[error("logit row {row_index} is empty")]
    EmptyLogitRow { row_index: usize },

    #[error("logit row {row_index} has label count {actual}, expected {expected}")]
    LabelCountMismatch {
        row_index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("token index {token_index} is out of range for token count {token_count}")]
    TokenIndexOutOfRange {
        token_index: usize,
        token_count: usize,
    },

    #[error("score contains non-finite value at row {row_index}, label {label_index}: {value}")]
    NonFiniteScore {
        row_index: usize,
        label_index: usize,
        value: f32,
    },

    #[error("aggregated token {token_index} has no scores")]
    MissingAggregatedScore { token_index: usize },
}

pub fn aggregate_window_logits(
    windows: &[WindowLogits],
    token_count: usize,
    label_count: usize,
) -> Result<TokenScores, ScoringError> {
    let mut sums: Vec<Option<Vec<f32>>> = vec![None; token_count];
    let mut counts = vec![0usize; token_count];

    for window in windows {
        if window.token_indices.len() != window.logits.len() {
            return Err(ScoringError::WindowLengthMismatch {
                token_indices: window.token_indices.len(),
                logits: window.logits.len(),
            });
        }

        let logprobs = log_softmax_rows(&window.logits, label_count)?;
        for (row_index, (token_index, row)) in window.token_indices.iter().zip(logprobs).enumerate()
        {
            if *token_index >= token_count {
                return Err(ScoringError::TokenIndexOutOfRange {
                    token_index: *token_index,
                    token_count,
                });
            }

            match &mut sums[*token_index] {
                Some(existing) => {
                    for (label_index, value) in row.iter().enumerate() {
                        existing[label_index] = logaddexp(existing[label_index], *value);
                    }
                }
                None => sums[*token_index] = Some(row),
            }
            counts[*token_index] += 1;
            let _ = row_index;
        }
    }

    let mut token_positions = Vec::new();
    let mut logprobs = Vec::new();

    for (token_index, maybe_sum) in sums.into_iter().enumerate() {
        let Some(mut row) = maybe_sum else {
            continue;
        };
        let count = counts[token_index];
        if count == 0 {
            return Err(ScoringError::MissingAggregatedScore { token_index });
        }
        let log_count = (count as f32).ln();
        for (label_index, value) in row.iter_mut().enumerate() {
            *value -= log_count;
            if !value.is_finite() {
                return Err(ScoringError::NonFiniteScore {
                    row_index: token_index,
                    label_index,
                    value: *value,
                });
            }
        }
        token_positions.push(token_index);
        logprobs.push(row);
    }

    Ok(TokenScores {
        token_positions,
        logprobs,
    })
}

pub fn log_softmax_rows(
    logits: &[Vec<f32>],
    label_count: usize,
) -> Result<Vec<Vec<f32>>, ScoringError> {
    logits
        .iter()
        .enumerate()
        .map(|(row_index, row)| log_softmax_row(row, row_index, label_count))
        .collect()
}

fn log_softmax_row(
    row: &[f32],
    row_index: usize,
    label_count: usize,
) -> Result<Vec<f32>, ScoringError> {
    if row.is_empty() {
        return Err(ScoringError::EmptyLogitRow { row_index });
    }
    if row.len() != label_count {
        return Err(ScoringError::LabelCountMismatch {
            row_index,
            expected: label_count,
            actual: row.len(),
        });
    }

    let mut max = f32::NEG_INFINITY;
    for (label_index, value) in row.iter().enumerate() {
        if !value.is_finite() {
            return Err(ScoringError::NonFiniteScore {
                row_index,
                label_index,
                value: *value,
            });
        }
        max = max.max(*value);
    }

    let sum_exp = row.iter().map(|value| (*value - max).exp()).sum::<f32>();
    let logsumexp = max + sum_exp.ln();
    let logprobs = row
        .iter()
        .enumerate()
        .map(|(label_index, value)| {
            let logprob = *value - logsumexp;
            if !logprob.is_finite() {
                return Err(ScoringError::NonFiniteScore {
                    row_index,
                    label_index,
                    value: logprob,
                });
            }
            Ok(logprob)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(logprobs)
}

fn logaddexp(left: f32, right: f32) -> f32 {
    if left == f32::NEG_INFINITY {
        return right;
    }
    if right == f32::NEG_INFINITY {
        return left;
    }
    let max = left.max(right);
    max + ((left - max).exp() + (right - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_is_stable_for_large_logits() {
        let rows = log_softmax_rows(&[vec![1000.0, 1001.0]], 2).unwrap();

        assert!((rows[0][0] - -1.3132616).abs() < 0.0001);
        assert!((rows[0][1] - -0.31326166).abs() < 0.0001);
    }

    #[test]
    fn aggregate_averages_duplicate_token_observations() {
        let windows = vec![
            WindowLogits {
                token_indices: vec![0],
                logits: vec![vec![0.0, 2.0]],
            },
            WindowLogits {
                token_indices: vec![0],
                logits: vec![vec![2.0, 0.0]],
            },
        ];

        let scores = aggregate_window_logits(&windows, 1, 2).unwrap();

        assert_eq!(scores.token_positions, vec![0]);
        assert!((scores.logprobs[0][0] - scores.logprobs[0][1]).abs() < 0.0001);
    }

    #[test]
    fn rejects_token_index_out_of_range() {
        let windows = vec![WindowLogits {
            token_indices: vec![1],
            logits: vec![vec![0.0, 1.0]],
        }];

        assert!(aggregate_window_logits(&windows, 1, 2).is_err());
    }
}
