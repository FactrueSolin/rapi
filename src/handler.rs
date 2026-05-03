use axum::body::Body;
use axum::body::to_bytes;
use axum::extract::{Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;

use crate::forwarder::Forwarder;
use crate::interceptor;
use crate::plugin::PluginRegistry;

#[derive(Clone)]
pub struct AppState {
    pub forwarder: Arc<Forwarder>,
    pub plugin_registry: Arc<PluginRegistry>,
}

pub async fn forward_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
    request: Request<Body>,
) -> Response {
    let target_str = headers
        .get("target")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .or_else(|| params.get("target").cloned());

    let target_str = match target_str {
        Some(url) => url,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "Missing required target: provide via 'target' header or query parameter",
            )
                .into_response();
        }
    };

    let target_base = match url::Url::parse(&target_str) {
        Ok(u) => u,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("Invalid target URL: {}", e),
            )
                .into_response();
        }
    };

    let (parts, body) = request.into_parts();
    let original_path = parts.uri.path();

    let mut filtered_query: Vec<(String, String)> = params
        .into_iter()
        .filter(|(k, _)| k != "target")
        .collect();
    filtered_query.sort_by_key(|(k, _)| k.clone());

    let query_string = if filtered_query.is_empty() {
        String::new()
    } else {
        let encoded: Vec<String> = filtered_query
            .iter()
            .map(|(k, v)| format!("{}={}", url::form_urlencoded::byte_serialize(k.as_bytes()).collect::<String>(), url::form_urlencoded::byte_serialize(v.as_bytes()).collect::<String>()))
            .collect();
        format!("?{}", encoded.join("&"))
    };

    let final_url = format!(
        "{}://{}{}{}",
        target_base.scheme(),
        target_base.authority(),
        original_path,
        query_string
    );

    let intercept_type = interceptor::should_intercept(original_path);

    if !matches!(intercept_type, interceptor::InterceptType::None) {
        let body_bytes = match to_bytes(body, usize::MAX).await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => {
                eprintln!("Failed to read body: {}", e);
                return (StatusCode::BAD_REQUEST, "Failed to read request body").into_response();
            }
        };

        let modified_body = match interceptor::intercept_body(&body_bytes, intercept_type, &state.plugin_registry).await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Interceptor error: {}", e);
                return (StatusCode::BAD_REQUEST, e).into_response();
            }
        };

        match state.forwarder.forward_with_bytes(&final_url, parts.method, parts.headers, modified_body).await {
            Ok(response) => response.into_response(),
            Err((status, message)) => {
                eprintln!("Forward error: {}", message);
                (status, message).into_response()
            }
        }
    } else {
        match state.forwarder.forward(&final_url, parts.method, parts.headers, body).await {
            Ok(response) => response.into_response(),
            Err((status, message)) => {
                eprintln!("Forward error: {}", message);
                (status, message).into_response()
            }
        }
    }
}
