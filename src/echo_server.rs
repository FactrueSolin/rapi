use axum::{
    extract::Request,
    response::IntoResponse,
    routing::any,
    Router,
    Json,
};
use http_body_util::BodyExt;
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;

#[derive(Serialize)]
struct EchoResponse {
    uri: String,
    headers: HashMap<String, String>,
    body: String,
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/{*path}", any(handle_request));

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8081);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Echo server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_request(request: Request) -> impl IntoResponse {
    let uri = request.uri().clone();
    let headers = request.headers().clone();

    let (_parts, body) = request.into_parts();
    let bytes = body
        .collect()
        .await
        .map(|collected| collected.to_bytes())
        .unwrap_or_default();

    let body_str = String::from_utf8_lossy(&bytes).to_string();

    let mut header_map = HashMap::new();
    for (key, value) in headers.iter() {
        header_map.insert(
            key.to_string(),
            value.to_str().unwrap_or("<binary>").to_string(),
        );
    }

    let response = EchoResponse {
        uri: uri.to_string(),
        headers: header_map,
        body: body_str.clone(),
    };

    println!("=== Incoming Request ===");
    println!("URI: {}", uri);
    println!("Headers:");
    for (key, value) in headers.iter() {
        println!("  {}: {:?}", key, value);
    }
    println!("Body:");
    println!("{}", body_str);
    println!("==========================\n");

    Json(response)
}
