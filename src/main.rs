mod forwarder;
mod handler;
mod interceptor;
mod plugin;
mod openaichatcompletion;

use axum::routing::any;
use axum::Router;
use handler::AppState;
use plugin::openai_privacy::OpenAiPrivacyPlugin;
use plugin::PluginRegistry;
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    let forwarder = Arc::new(forwarder::Forwarder::new());

    let mut plugin_registry = PluginRegistry::new();
    plugin_registry.register(Arc::new(OpenAiPrivacyPlugin::from_env()));
    let plugin_registry = Arc::new(plugin_registry);

    let state = AppState {
        forwarder,
        plugin_registry,
    };

    let app = Router::new()
        .route("/{*path}", any(handler::forward_handler))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Transparent forwarder listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app)
        .await
        .expect("Server failed");
}
