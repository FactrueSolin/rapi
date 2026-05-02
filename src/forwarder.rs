use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, Method, Response, StatusCode};
use reqwest::Client;

static HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

pub struct Forwarder {
    client: Client,
}

impl Forwarder {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    pub async fn forward(
        &self,
        target_url: &str,
        method: Method,
        headers: HeaderMap,
        body: Body,
    ) -> Result<Response<Body>, (StatusCode, String)> {
        let parsed_url = url::Url::parse(target_url)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid target URL: {}", e)))?;

        let mut request_builder = self.client.request(method.clone(), parsed_url);

        for (name, value) in headers.iter() {
            let name_lower = name.as_str().to_lowercase();
            if HOP_BY_HOP_HEADERS.contains(&name_lower.as_str()) || name_lower == "host" {
                continue;
            }
            if let Ok(header_name) = HeaderName::from_bytes(name.as_ref()) {
                request_builder = request_builder.header(header_name, value.clone());
            }
        }

        let hyper_body = body;
        let reqwest_body = reqwest::Body::wrap_stream(hyper_body.into_data_stream());
        request_builder = request_builder.body(reqwest_body);

        let response = request_builder
            .send()
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Forward request failed: {}", e)))?;

        let status = StatusCode::from_u16(response.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);

        let mut response_headers = HeaderMap::new();
        for (name, value) in response.headers().iter() {
            let name_lower = name.as_str().to_lowercase();
            if HOP_BY_HOP_HEADERS.contains(&name_lower.as_str()) {
                continue;
            }
            if let Ok(header_name) = HeaderName::from_bytes(name.as_ref()) {
                response_headers.insert(header_name, value.clone());
            }
        }

        let response_body = Body::from_stream(response.bytes_stream());

        let mut response = Response::builder()
            .status(status)
            .body(response_body)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Build response failed: {}", e)))?;

        *response.headers_mut() = response_headers;

        Ok(response)
    }
}
