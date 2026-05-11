use std::time::Duration;

use reqwest::Client;

use super::model_files::{DEFAULT_REVISION, MODEL_REPOSITORY};
use super::model_manager::ModelManagerError;

const OFFICIAL_ENDPOINT: &str = "https://huggingface.co";
const MIRROR_ENDPOINT: &str = "https://hf-mirror.com";
const PROBE_FILE: &str = "config.json";
const OFFICIAL_TIMEOUT_SECS: u64 = 20;
const MIRROR_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointSource {
    Official,
    Mirror,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEndpoint {
    pub source: EndpointSource,
    pub base_url: String,
}

#[derive(Debug, Clone)]
pub struct EndpointProbe {
    official_endpoint: String,
    mirror_endpoint: String,
    official_timeout: Duration,
    mirror_timeout: Duration,
    client: Client,
}

impl Default for EndpointProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointProbe {
    pub fn new() -> Self {
        Self {
            official_endpoint: OFFICIAL_ENDPOINT.to_string(),
            mirror_endpoint: MIRROR_ENDPOINT.to_string(),
            official_timeout: Duration::from_secs(OFFICIAL_TIMEOUT_SECS),
            mirror_timeout: Duration::from_secs(MIRROR_TIMEOUT_SECS),
            client: Client::new(),
        }
    }

    #[cfg(test)]
    pub fn with_endpoints(
        official_endpoint: impl Into<String>,
        mirror_endpoint: impl Into<String>,
        official_timeout: Duration,
        mirror_timeout: Duration,
    ) -> Self {
        Self {
            official_endpoint: official_endpoint.into(),
            mirror_endpoint: mirror_endpoint.into(),
            official_timeout,
            mirror_timeout,
            client: Client::new(),
        }
    }

    pub async fn resolve(&self) -> Result<ResolvedEndpoint, ModelManagerError> {
        match self
            .probe(&self.official_endpoint, self.official_timeout)
            .await
        {
            Ok(()) => Ok(ResolvedEndpoint {
                source: EndpointSource::Official,
                base_url: self.official_endpoint.clone(),
            }),
            Err(official_error) => {
                let official_error_message = official_error.to_string();
                match self.probe(&self.mirror_endpoint, self.mirror_timeout).await {
                    Ok(()) => Ok(ResolvedEndpoint {
                        source: EndpointSource::Mirror,
                        base_url: self.mirror_endpoint.clone(),
                    }),
                    Err(mirror_error) => Err(ModelManagerError::NoEndpointAvailable {
                        official_error: official_error_message,
                        mirror_error: mirror_error.to_string(),
                    }),
                }
            }
        }
    }

    async fn probe(&self, endpoint: &str, timeout: Duration) -> Result<(), reqwest::Error> {
        let url = probe_url(endpoint);
        self.client
            .get(url)
            .timeout(timeout)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

fn probe_url(endpoint: &str) -> String {
    format!(
        "{}/{}/resolve/{}/{}",
        endpoint.trim_end_matches('/'),
        MODEL_REPOSITORY,
        DEFAULT_REVISION,
        PROBE_FILE
    )
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    use super::*;

    fn spawn_probe_server(status: u16) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("test server address");
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                drain_request(&mut stream);
                let reason = if status == 200 { "OK" } else { "ERROR" };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{address}")
    }

    fn drain_request(stream: &mut TcpStream) {
        let mut buffer = [0_u8; 1024];
        let _ = stream.read(&mut buffer);
    }

    #[test]
    fn probe_url_uses_resolve_get_target() {
        assert_eq!(
            probe_url("https://huggingface.co/"),
            "https://huggingface.co/openai/privacy-filter/resolve/main/config.json"
        );
    }

    #[tokio::test]
    async fn selects_official_when_official_succeeds() {
        let official = spawn_probe_server(200);
        let mirror = spawn_probe_server(200);
        let probe = EndpointProbe::with_endpoints(
            official.clone(),
            mirror,
            Duration::from_secs(2),
            Duration::from_secs(2),
        );

        let endpoint = probe.resolve().await.expect("resolve endpoint");

        assert_eq!(endpoint.source, EndpointSource::Official);
        assert_eq!(endpoint.base_url, official);
    }

    #[tokio::test]
    async fn falls_back_to_mirror_when_official_fails() {
        let official = spawn_probe_server(500);
        let mirror = spawn_probe_server(200);
        let probe = EndpointProbe::with_endpoints(
            official,
            mirror.clone(),
            Duration::from_secs(2),
            Duration::from_secs(2),
        );

        let endpoint = probe.resolve().await.expect("resolve endpoint");

        assert_eq!(endpoint.source, EndpointSource::Mirror);
        assert_eq!(endpoint.base_url, mirror);
    }

    #[tokio::test]
    async fn errors_when_both_endpoints_fail() {
        let official = spawn_probe_server(500);
        let mirror = spawn_probe_server(500);
        let probe = EndpointProbe::with_endpoints(
            official,
            mirror,
            Duration::from_secs(2),
            Duration::from_secs(2),
        );

        let error = probe.resolve().await.expect_err("resolve should fail");

        assert!(matches!(
            error,
            ModelManagerError::NoEndpointAvailable { .. }
        ));
    }
}
