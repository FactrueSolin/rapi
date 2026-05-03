use std::sync::Arc;

use crate::anthropicmessage;
use crate::openaichatcompletion;
use crate::plugin::types::PluginMessageView;
use crate::plugin::PluginRegistry;

pub enum InterceptType {
    OpenAI,
    Anthropic,
    None,
}

pub fn should_intercept(path: &str) -> InterceptType {
    if path.ends_with("/chat/completions") {
        InterceptType::OpenAI
    } else if path.ends_with("/v1/messages") {
        InterceptType::Anthropic
    } else {
        InterceptType::None
    }
}

pub async fn intercept_body(
    body: &[u8],
    intercept_type: InterceptType,
    registry: &Arc<PluginRegistry>,
) -> Result<Vec<u8>, String> {
    let json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| format!("Failed to parse request body as JSON: {}", e))?;

    match intercept_type {
        InterceptType::OpenAI => {
            let extracted = openaichatcompletion::extract_text_messages(json)?;
            let modified = registry.process_messages(extracted).await?;
            let modified_body = serde_json::to_vec(&modified.into_body())
                .map_err(|e| format!("Failed to serialize modified body: {}", e))?;
            Ok(modified_body)
        }
        InterceptType::Anthropic => {
            let extracted = anthropicmessage::extract_text_messages(json)?;
            let view = PluginMessageView::from_anthropic_extracted(&extracted);
            let modified = registry.process_anthropic_messages(extracted, &view).await?;
            let modified_body = serde_json::to_vec(&modified.into_body())
                .map_err(|e| format!("Failed to serialize modified body: {}", e))?;
            Ok(modified_body)
        }
        InterceptType::None => {
            Err("No intercept type specified".to_string())
        }
    }
}
