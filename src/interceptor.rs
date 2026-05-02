use std::sync::Arc;

use crate::openaichatcompletion;
use crate::plugin::PluginRegistry;

pub fn should_intercept(path: &str) -> bool {
    path.ends_with("/chat/completions")
}

pub async fn intercept_body(
    body: &[u8],
    registry: &Arc<PluginRegistry>,
) -> Result<Vec<u8>, String> {
    let json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| format!("Failed to parse request body as JSON: {}", e))?;

    let extracted = openaichatcompletion::extract_text_messages(json)?;

    let modified = registry.process_messages(extracted).await?;

    let modified_body = serde_json::to_vec(&modified.into_body())
        .map_err(|e| format!("Failed to serialize modified body: {}", e))?;

    Ok(modified_body)
}
