use axum::body::Body;
use axum::extract::{Query, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;

use crate::forwarder::Forwarder;

#[derive(Clone)]
pub struct AppState {
    pub forwarder: std::sync::Arc<Forwarder>,
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

    match state.forwarder.forward(&target_url, parts.method, parts.headers, body).await {
        Ok(response) => response.into_response(),
        Err((status, message)) => {
            eprintln!("Forward error: {}", message);
            (status, message).into_response()
        }
    }
}
