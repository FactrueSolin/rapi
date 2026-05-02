use async_trait::async_trait;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

use super::{Plugin, PluginMessageView, PluginResult};
use super::types::{MessageReplacements, Replacement};

#[derive(Error, Debug)]
pub enum OpenAiPrivacyError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON parse failed: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub model_loaded: bool,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplacementPair {
    pub original: String,
    pub replacement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactResult {
    pub pairs: Vec<ReplacementPair>,
    pub redacted_text: String,
}

pub struct OpenAiPrivacyClient {
    client: reqwest::Client,
    base_url: String,
}

impl OpenAiPrivacyClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        let base_url = std::env::var("OPENAI_PRIVACY_URL")
            .unwrap_or_else(|_| "http://localhost:8000".to_string());
        Self::new(base_url)
    }

    pub async fn health_check(&self) -> Result<HealthStatus, OpenAiPrivacyError> {
        let url = format!("{}/health", self.base_url);
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OpenAiPrivacyError::ServiceUnavailable(format!(
                "Health check failed with status {}: {}",
                status, body
            )));
        }

        let health: HealthStatus = response.json().await?;
        Ok(health)
    }

    pub async fn redact(&self, text: &str) -> Result<RedactResult, OpenAiPrivacyError> {
        let url = format!("{}/redact", self.base_url);

        #[derive(Serialize)]
        struct RedactRequest<'a> {
            text: &'a str,
        }

        let response = self
            .client
            .post(&url)
            .json(&RedactRequest { text })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OpenAiPrivacyError::ServiceUnavailable(format!(
                "Redact request failed with status {}: {}",
                status, body
            )));
        }

        let result: RedactResult = response.json().await?;
        Ok(result)
    }

    pub async fn redact_concurrent(
        &self,
        texts: &[String],
    ) -> Vec<Result<RedactResult, OpenAiPrivacyError>> {
        let futures: Vec<_> = texts
            .iter()
            .map(|text| self.redact(text))
            .collect();
        join_all(futures).await
    }
}

pub struct OpenAiPrivacyPlugin {
    client: OpenAiPrivacyClient,
}

impl OpenAiPrivacyPlugin {
    pub fn new(client: OpenAiPrivacyClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> Self {
        Self::new(OpenAiPrivacyClient::from_env())
    }
}

#[async_trait]
impl Plugin for OpenAiPrivacyPlugin {
    fn name(&self) -> &str {
        "openai_privacy"
    }

    async fn process(&self, messages: &PluginMessageView) -> Result<PluginResult, String> {
        let entries = messages.entries();
        if entries.is_empty() {
            return Ok(PluginResult {
                message_replacements: Vec::new(),
            });
        }

        let start = Instant::now();
        let texts: Vec<String> = entries.iter().map(|e| e.text().to_string()).collect();
        let results = self.client.redact_concurrent(&texts).await;

        let mut message_replacements = Vec::new();
        let mut success_count = 0;
        let mut error_count = 0;

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(redact_result) => {
                    let entry = &entries[i];
                    message_replacements.push(MessageReplacements {
                        message_index: entry.index(),
                        replacements: vec![Replacement {
                            original: entry.text().to_string(),
                            replacement: redact_result.redacted_text,
                        }],
                    });
                    success_count += 1;
                }
                Err(e) => {
                    error_count += 1;
                    let error_type = match &e {
                        OpenAiPrivacyError::HttpError(http_err) => {
                            if http_err.is_timeout() {
                                "timeout"
                            } else if http_err.is_connect() {
                                "connection_failed"
                            } else {
                                "http_error"
                            }
                        }
                        OpenAiPrivacyError::ServiceUnavailable(_) => "service_unavailable",
                        OpenAiPrivacyError::JsonError(_) => "parse_error",
                    };
                    eprintln!(
                        "[openai_privacy] {} on message {} (index {}): {}",
                        error_type, i, entries[i].index(), e
                    );
                }
            }
        }

        let elapsed = start.elapsed();
        eprintln!(
            "[openai_privacy] Processed {} messages: {} succeeded, {} failed, elapsed={:?}",
            entries.len(),
            success_count,
            error_count,
            elapsed
        );

        Ok(PluginResult {
            message_replacements,
        })
    }
}
