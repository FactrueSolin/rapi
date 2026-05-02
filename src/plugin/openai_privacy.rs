use futures::future::join_all;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
        Self {
            client: reqwest::Client::new(),
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
