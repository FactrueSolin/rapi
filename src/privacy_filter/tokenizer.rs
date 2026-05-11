use std::path::PathBuf;

use serde::Deserialize;
use thiserror::Error;
use tokenizers::Tokenizer;

use super::model_manager::ResolvedModelPaths;

const DEFAULT_CONTEXT_WINDOW_LENGTH: usize = 4096;
const INPUT_IDS_NAME: &str = "input_ids";
const ATTENTION_MASK_NAME: &str = "attention_mask";

#[derive(Debug, Clone, Default)]
pub struct TokenizerOptions {
    pub context_window_length: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerRuntimeConfig {
    pub model_input_names: Vec<String>,
    pub model_max_length: Option<usize>,
    pub context_window_length: usize,
    pub pad_token_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedText {
    pub original_text: String,
    pub token_ids: Vec<u32>,
    pub token_offsets: Vec<TokenOffset>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenOffset {
    pub token_index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenWindow {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PrivacyFilterTokenizer {
    tokenizer: Tokenizer,
    config: TokenizerRuntimeConfig,
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("tokenizer load failed for {path}: {source}")]
    Load {
        path: PathBuf,
        source: tokenizers::Error,
    },

    #[error("tokenizer config read failed for {path}: {source}")]
    ConfigRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("tokenizer config parse failed for {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("tokenization failed: {source}")]
    Encode { source: tokenizers::Error },

    #[error("token offset count mismatch: tokens={tokens}, offsets={offsets}")]
    OffsetCountMismatch { tokens: usize, offsets: usize },

    #[error("invalid token offset for token {token_index}: {start_byte}..{end_byte}")]
    InvalidOffset {
        token_index: usize,
        start_byte: usize,
        end_byte: usize,
    },

    #[error("invalid context window length: {0}")]
    InvalidContextWindowLength(usize),

    #[error("tokenizer config missing required model input: {0}")]
    MissingModelInput(String),
}

impl PrivacyFilterTokenizer {
    pub fn from_paths(
        paths: &ResolvedModelPaths,
        options: TokenizerOptions,
    ) -> Result<Self, TokenizerError> {
        let tokenizer =
            Tokenizer::from_file(&paths.tokenizer_path).map_err(|source| TokenizerError::Load {
                path: paths.tokenizer_path.clone(),
                source,
            })?;

        let tokenizer_config = read_tokenizer_config(&paths.tokenizer_config_path)?;
        let model_config = read_model_tokenizer_config(&paths.config_path)?;
        let config = build_runtime_config(tokenizer_config, model_config, options)?;

        Ok(Self { tokenizer, config })
    }

    #[cfg(test)]
    fn from_tokenizer_for_test(
        tokenizer: Tokenizer,
        config: TokenizerRuntimeConfig,
    ) -> Result<Self, TokenizerError> {
        validate_context_window_length(config.context_window_length)?;
        validate_model_inputs(&config.model_input_names)?;
        Ok(Self { tokenizer, config })
    }

    pub fn config(&self) -> &TokenizerRuntimeConfig {
        &self.config
    }

    pub fn encode(&self, text: &str) -> Result<TokenizedText, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|source| TokenizerError::Encode { source })?;
        let token_ids = encoding.get_ids().to_vec();
        let raw_offsets = encoding.get_offsets();

        if token_ids.len() != raw_offsets.len() {
            return Err(TokenizerError::OffsetCountMismatch {
                tokens: token_ids.len(),
                offsets: raw_offsets.len(),
            });
        }

        let token_offsets = raw_offsets
            .iter()
            .enumerate()
            .map(|(token_index, (start_byte, end_byte))| {
                validate_offset(text, token_index, *start_byte, *end_byte)?;
                Ok(TokenOffset {
                    token_index,
                    start_byte: *start_byte,
                    end_byte: *end_byte,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(TokenizedText {
            original_text: text.to_string(),
            token_ids,
            token_offsets,
        })
    }

    pub fn windows(&self, tokenized: &TokenizedText) -> Result<Vec<TokenWindow>, TokenizerError> {
        validate_context_window_length(self.config.context_window_length)?;

        Ok(windows_for_token_ids(
            &tokenized.token_ids,
            self.config.context_window_length,
        ))
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    model_input_names: Option<Vec<String>>,
    model_max_length: Option<usize>,
    #[allow(dead_code)]
    tokenizer_class: Option<String>,
    #[allow(dead_code)]
    backend: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelTokenizerConfigFile {
    pad_token_id: Option<i64>,
    default_n_ctx: Option<usize>,
    initial_context_length: Option<usize>,
    #[allow(dead_code)]
    max_position_embeddings: Option<usize>,
}

fn read_tokenizer_config(path: &PathBuf) -> Result<TokenizerConfigFile, TokenizerError> {
    let content = std::fs::read_to_string(path).map_err(|source| TokenizerError::ConfigRead {
        path: path.clone(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| TokenizerError::ConfigParse {
        path: path.clone(),
        source,
    })
}

fn read_model_tokenizer_config(path: &PathBuf) -> Result<ModelTokenizerConfigFile, TokenizerError> {
    let content = std::fs::read_to_string(path).map_err(|source| TokenizerError::ConfigRead {
        path: path.clone(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| TokenizerError::ConfigParse {
        path: path.clone(),
        source,
    })
}

fn build_runtime_config(
    tokenizer_config: TokenizerConfigFile,
    model_config: ModelTokenizerConfigFile,
    options: TokenizerOptions,
) -> Result<TokenizerRuntimeConfig, TokenizerError> {
    let model_input_names = tokenizer_config.model_input_names.unwrap_or_default();
    validate_model_inputs(&model_input_names)?;

    let context_window_length = options
        .context_window_length
        .or(model_config.initial_context_length)
        .or(model_config.default_n_ctx)
        .or(tokenizer_config.model_max_length)
        .unwrap_or(DEFAULT_CONTEXT_WINDOW_LENGTH);
    validate_context_window_length(context_window_length)?;

    Ok(TokenizerRuntimeConfig {
        model_input_names,
        model_max_length: tokenizer_config.model_max_length,
        context_window_length,
        pad_token_id: model_config.pad_token_id,
    })
}

fn validate_model_inputs(model_input_names: &[String]) -> Result<(), TokenizerError> {
    if !model_input_names.iter().any(|name| name == INPUT_IDS_NAME) {
        return Err(TokenizerError::MissingModelInput(
            INPUT_IDS_NAME.to_string(),
        ));
    }

    if !model_input_names
        .iter()
        .any(|name| name == ATTENTION_MASK_NAME)
    {
        return Err(TokenizerError::MissingModelInput(
            ATTENTION_MASK_NAME.to_string(),
        ));
    }

    Ok(())
}

fn validate_context_window_length(context_window_length: usize) -> Result<(), TokenizerError> {
    if context_window_length == 0 {
        return Err(TokenizerError::InvalidContextWindowLength(
            context_window_length,
        ));
    }

    Ok(())
}

fn validate_offset(
    text: &str,
    token_index: usize,
    start_byte: usize,
    end_byte: usize,
) -> Result<(), TokenizerError> {
    if start_byte > end_byte
        || end_byte > text.len()
        || !text.is_char_boundary(start_byte)
        || !text.is_char_boundary(end_byte)
    {
        return Err(TokenizerError::InvalidOffset {
            token_index,
            start_byte,
            end_byte,
        });
    }

    Ok(())
}

fn windows_for_token_ids(token_ids: &[u32], context_window_length: usize) -> Vec<TokenWindow> {
    token_ids
        .chunks(context_window_length)
        .enumerate()
        .map(|(window_index, input_ids)| {
            let start_index = window_index * context_window_length;
            let token_indices = (start_index..start_index + input_ids.len()).collect();

            TokenWindow {
                input_ids: input_ids.to_vec(),
                attention_mask: vec![1; input_ids.len()],
                token_indices,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    fn temp_dir(test_name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("rapi-tokenizer-{test_name}-{suffix}"))
    }

    fn valid_input_names() -> Vec<String> {
        vec![INPUT_IDS_NAME.to_string(), ATTENTION_MASK_NAME.to_string()]
    }

    fn runtime_config(context_window_length: usize) -> TokenizerRuntimeConfig {
        TokenizerRuntimeConfig {
            model_input_names: valid_input_names(),
            model_max_length: Some(128000),
            context_window_length,
            pad_token_id: Some(199999),
        }
    }

    fn test_tokenizer() -> Tokenizer {
        let vocab = [
            ("[UNK]".to_string(), 0),
            ("hello".to_string(), 1),
            ("world".to_string(), 2),
            ("中文".to_string(), 3),
            ("emoji".to_string(), 4),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

    #[test]
    fn parses_minimal_tokenizer_config() {
        let tokenizer_config = TokenizerConfigFile {
            model_input_names: Some(valid_input_names()),
            model_max_length: Some(128000),
            tokenizer_class: None,
            backend: None,
        };
        let model_config = ModelTokenizerConfigFile {
            pad_token_id: Some(199999),
            default_n_ctx: Some(8192),
            initial_context_length: Some(4096),
            max_position_embeddings: None,
        };

        let config =
            build_runtime_config(tokenizer_config, model_config, TokenizerOptions::default())
                .expect("runtime config");

        assert_eq!(config.model_input_names, valid_input_names());
        assert_eq!(config.model_max_length, Some(128000));
        assert_eq!(config.context_window_length, 4096);
        assert_eq!(config.pad_token_id, Some(199999));
    }

    #[test]
    fn tokenizer_options_override_context_window_length() {
        let tokenizer_config = TokenizerConfigFile {
            model_input_names: Some(valid_input_names()),
            model_max_length: Some(128000),
            tokenizer_class: None,
            backend: None,
        };
        let model_config = ModelTokenizerConfigFile {
            pad_token_id: None,
            default_n_ctx: Some(8192),
            initial_context_length: Some(4096),
            max_position_embeddings: None,
        };

        let config = build_runtime_config(
            tokenizer_config,
            model_config,
            TokenizerOptions {
                context_window_length: Some(128),
            },
        )
        .expect("runtime config");

        assert_eq!(config.context_window_length, 128);
    }

    #[test]
    fn falls_back_to_default_context_window_length() {
        let tokenizer_config = TokenizerConfigFile {
            model_input_names: Some(valid_input_names()),
            model_max_length: None,
            tokenizer_class: None,
            backend: None,
        };
        let model_config = ModelTokenizerConfigFile {
            pad_token_id: None,
            default_n_ctx: None,
            initial_context_length: None,
            max_position_embeddings: None,
        };

        let config =
            build_runtime_config(tokenizer_config, model_config, TokenizerOptions::default())
                .expect("runtime config");

        assert_eq!(config.context_window_length, DEFAULT_CONTEXT_WINDOW_LENGTH);
    }

    #[test]
    fn missing_input_ids_returns_error() {
        let input_names = vec![ATTENTION_MASK_NAME.to_string()];

        let error = validate_model_inputs(&input_names).expect_err("missing input_ids");

        assert!(matches!(
            error,
            TokenizerError::MissingModelInput(name) if name == INPUT_IDS_NAME
        ));
    }

    #[test]
    fn missing_attention_mask_returns_error() {
        let input_names = vec![INPUT_IDS_NAME.to_string()];

        let error = validate_model_inputs(&input_names).expect_err("missing attention_mask");

        assert!(matches!(
            error,
            TokenizerError::MissingModelInput(name) if name == ATTENTION_MASK_NAME
        ));
    }

    #[test]
    fn zero_context_window_length_returns_error() {
        let error = validate_context_window_length(0).expect_err("zero context window");

        assert!(matches!(
            error,
            TokenizerError::InvalidContextWindowLength(0)
        ));
    }

    #[test]
    fn windows_split_tokens_without_overlap() {
        let tokenizer =
            PrivacyFilterTokenizer::from_tokenizer_for_test(test_tokenizer(), runtime_config(3))
                .expect("tokenizer");
        let tokenized = TokenizedText {
            original_text: "".to_string(),
            token_ids: vec![10, 11, 12, 13, 14, 15, 16],
            token_offsets: Vec::new(),
        };

        let windows = tokenizer.windows(&tokenized).expect("windows");

        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].input_ids, vec![10, 11, 12]);
        assert_eq!(windows[0].attention_mask, vec![1, 1, 1]);
        assert_eq!(windows[0].token_indices, vec![0, 1, 2]);
        assert_eq!(windows[1].input_ids, vec![13, 14, 15]);
        assert_eq!(windows[1].attention_mask, vec![1, 1, 1]);
        assert_eq!(windows[1].token_indices, vec![3, 4, 5]);
        assert_eq!(windows[2].input_ids, vec![16]);
        assert_eq!(windows[2].attention_mask, vec![1]);
        assert_eq!(windows[2].token_indices, vec![6]);
    }

    #[test]
    fn empty_text_returns_empty_tokens_and_windows() {
        let tokenizer =
            PrivacyFilterTokenizer::from_tokenizer_for_test(test_tokenizer(), runtime_config(3))
                .expect("tokenizer");

        let tokenized = tokenizer.encode("").expect("encode");
        let windows = tokenizer.windows(&tokenized).expect("windows");

        assert_eq!(tokenized.original_text, "");
        assert!(tokenized.token_ids.is_empty());
        assert!(tokenized.token_offsets.is_empty());
        assert!(windows.is_empty());
    }

    #[test]
    fn utf8_offsets_are_valid_byte_boundaries() {
        let tokenizer =
            PrivacyFilterTokenizer::from_tokenizer_for_test(test_tokenizer(), runtime_config(4))
                .expect("tokenizer");
        let text = "hello 中文 emoji";

        let tokenized = tokenizer.encode(text).expect("encode");

        assert!(!tokenized.token_ids.is_empty());
        for offset in tokenized.token_offsets {
            assert!(text.is_char_boundary(offset.start_byte));
            assert!(text.is_char_boundary(offset.end_byte));
            assert!(offset.end_byte <= text.len());
            let _ = &text[offset.start_byte..offset.end_byte];
        }
    }

    #[test]
    fn invalid_utf8_boundary_offset_returns_error() {
        let text = "中文";

        let error = validate_offset(text, 0, 1, 3).expect_err("invalid boundary");

        assert!(matches!(
            error,
            TokenizerError::InvalidOffset {
                token_index: 0,
                start_byte: 1,
                end_byte: 3
            }
        ));
    }

    #[test]
    fn from_paths_loads_tokenizer_and_configs() {
        let dir = temp_dir("from-paths");
        fs::create_dir_all(&dir).expect("create temp dir");
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer_config_path = dir.join("tokenizer_config.json");
        let config_path = dir.join("config.json");
        test_tokenizer()
            .save(&tokenizer_path, false)
            .expect("save tokenizer");
        fs::write(
            &tokenizer_config_path,
            r#"{"model_input_names":["input_ids","attention_mask"],"model_max_length":128000}"#,
        )
        .expect("write tokenizer config");
        fs::write(
            &config_path,
            r#"{"pad_token_id":199999,"default_n_ctx":8192,"initial_context_length":4096}"#,
        )
        .expect("write config");
        let paths = ResolvedModelPaths {
            endpoint: super::super::endpoint_probe::ResolvedEndpoint {
                source: super::super::endpoint_probe::EndpointSource::Official,
                base_url: "https://huggingface.co".to_string(),
            },
            cache_dir: dir.clone(),
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            viterbi_calibration_path: dir.join("viterbi_calibration.json"),
            model_path: dir.join("onnx/model_quantized.onnx"),
            model_data_paths: Vec::new(),
        };

        let tokenizer = PrivacyFilterTokenizer::from_paths(&paths, TokenizerOptions::default())
            .expect("load tokenizer");

        assert_eq!(tokenizer.config().context_window_length, 4096);
        assert_eq!(tokenizer.config().pad_token_id, Some(199999));

        let _ = fs::remove_dir_all(dir);
    }
}
