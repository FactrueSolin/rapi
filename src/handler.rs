use axum::body::Body;
use axum::body::to_bytes;
use axum::extract::{Query, Request, State};
use axum::http::StatusCode;
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
    Query(params): Query<HashMap<String, String>>,
    request: Request<Body>,
) -> Response {
    let target_url = match params.get("target") {
        Some(url) => url.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "Missing required query parameter: target",
            )
                .into_response();
        }
    };

    let (parts, body) = request.into_parts();
    let path = parts.uri.path().to_string();

    if interceptor::should_intercept(&path) {
        let body_bytes = match to_bytes(body, usize::MAX).await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => {
                eprintln!("Failed to read body: {}", e);
                return (StatusCode::BAD_REQUEST, "Failed to read request body").into_response();
            }
        };

        let modified_body = match interceptor::intercept_body(&body_bytes, &state.plugin_registry).await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Interceptor error: {}", e);
                return (StatusCode::BAD_REQUEST, e).into_response();
            }
        };

        match state.forwarder.forward_with_bytes(&target_url, parts.method, parts.headers, modified_body).await {
            Ok(response) => response.into_response(),
            Err((status, message)) => {
                eprintln!("Forward error: {}", message);
                (status, message).into_response()
            }
        }
    } else {
        match state.forwarder.forward(&target_url, parts.method, parts.headers, body).await {
            Ok(response) => response.into_response(),
            Err((status, message)) => {
                eprintln!("Forward error: {}", message);
                (status, message).into_response()
            }
        }
    }
}
