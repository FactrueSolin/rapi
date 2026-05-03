use serde_json::Value;

pub mod types;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MessageSource {
    System,
    Messages(usize),
}

#[derive(Clone, Debug)]
pub struct MessageRef {
    source: MessageSource,
}

impl MessageRef {
    pub fn system() -> Self {
        Self {
            source: MessageSource::System,
        }
    }

    pub fn messages(index: usize) -> Self {
        Self {
            source: MessageSource::Messages(index),
        }
    }

    pub fn from_source(source: MessageSource) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &MessageSource {
        &self.source
    }
}

pub struct ExtractedMessages {
    pub system: Option<MessageRef>,
    pub user: Vec<MessageRef>,
    pub assistant: Vec<MessageRef>,
    body: Value,
}

fn get_text_from_content(content: &Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }

    if let Some(parts) = content.as_array() {
        for part in parts {
            if part.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }
        }
    }

    None
}

fn get_text_from_system(system: &Value) -> Option<String> {
    if let Some(text) = system.as_str() {
        return Some(text.to_string());
    }

    if let Some(blocks) = system.as_array() {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }
        }
    }

    None
}

fn set_text_to_content(content: &mut Value, text: String) {
    if content.is_string() {
        *content = Value::String(text);
        return;
    }

    if let Some(parts) = content.as_array_mut() {
        for part in parts.iter_mut() {
            if part.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text_field) = part.get_mut("text") {
                    *text_field = Value::String(text);
                    return;
                }
            }
        }
        parts.push(serde_json::json!({
            "type": "text",
            "text": text
        }));
        return;
    }

    *content = Value::String(text);
}

fn set_text_to_system(system: &mut Value, text: String) {
    if system.is_string() {
        *system = Value::String(text);
        return;
    }

    if let Some(blocks) = system.as_array_mut() {
        for block in blocks.iter_mut() {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text_field) = block.get_mut("text") {
                    *text_field = Value::String(text);
                    return;
                }
            }
        }
        blocks.push(serde_json::json!({
            "type": "text",
            "text": text
        }));
        return;
    }

    *system = Value::String(text);
}

pub fn extract_text_messages(body: Value) -> Result<ExtractedMessages, String> {
    let mut result = ExtractedMessages {
        system: None,
        user: Vec::new(),
        assistant: Vec::new(),
        body,
    };

    if let Some(system) = result.body.get("system") {
        if !system.is_null() && get_text_from_system(system).is_some() {
            result.system = Some(MessageRef::system());
        }
    }

    let messages: Vec<(usize, Value)> = result
        .body
        .get("messages")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .enumerate()
                .map(|(i, msg)| (i, msg.clone()))
                .collect()
        })
        .unwrap_or_default();

    for (i, msg) in messages {
        let role = match msg.get("role").and_then(Value::as_str) {
            Some(r) => r,
            None => continue,
        };

        let content = match msg.get("content") {
            Some(c) => c,
            None => continue,
        };

        if content.is_null() {
            continue;
        }

        if get_text_from_content(content).is_some() {
            let msg_ref = MessageRef::messages(i);
            match role {
                "user" => result.user.push(msg_ref),
                "assistant" => result.assistant.push(msg_ref),
                _ => {}
            }
        }
    }

    Ok(result)
}

impl ExtractedMessages {
    pub fn get_text(&self, msg: &MessageRef) -> Option<String> {
        match msg.source() {
            MessageSource::System => {
                let system = self.body.get("system")?;
                if system.is_null() {
                    return None;
                }
                get_text_from_system(system)
            }
            MessageSource::Messages(index) => {
                let content = self
                    .body
                    .get("messages")
                    .and_then(Value::as_array)?
                    .get(*index)
                    .and_then(|m| m.get("content"))?;

                if content.is_null() {
                    return None;
                }

                get_text_from_content(content)
            }
        }
    }

    pub fn set_text(&mut self, msg: &MessageRef, text: String) {
        match msg.source() {
            MessageSource::System => {
                if let Some(system) = self.body.get_mut("system") {
                    set_text_to_system(system, text);
                }
            }
            MessageSource::Messages(index) => {
                if let Some(messages) = self.body.get_mut("messages").and_then(Value::as_array_mut) {
                    if let Some(message) = messages.get_mut(*index) {
                        if let Some(content) = message.get_mut("content") {
                            set_text_to_content(content, text);
                        }
                    }
                }
            }
        }
    }

    pub fn has_system(&self) -> bool {
        self.body.get("system").is_some_and(|s| !s.is_null())
    }

    pub fn into_body(self) -> Value {
        self.body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_string_messages() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [
                { "role": "user", "content": "Hello!" },
                { "role": "assistant", "content": "Hi there!" },
                { "role": "user", "content": "How are you?" }
            ]
        });

        let extracted = extract_text_messages(body).unwrap();
        assert!(extracted.system.is_some());
        assert_eq!(extracted.user.len(), 2);
        assert_eq!(extracted.assistant.len(), 1);

        if let Some(ref sys_ref) = extracted.system {
            assert_eq!(extracted.get_text(sys_ref).unwrap(), "You are a helpful assistant.");
        }
    }

    #[test]
    fn test_extract_content_blocks() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "What is this?" },
                        { "type": "image", "source": { "type": "base64", "data": "..." } }
                    ]
                }
            ]
        });

        let extracted = extract_text_messages(body).unwrap();
        assert_eq!(extracted.user.len(), 1);

        let user_ref = &extracted.user[0];
        assert_eq!(extracted.get_text(user_ref).unwrap(), "What is this?");
    }

    #[test]
    fn test_extract_system_as_blocks() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                { "type": "text", "text": "System instruction." }
            ],
            "messages": [
                { "role": "user", "content": "Hello" }
            ]
        });

        let extracted = extract_text_messages(body).unwrap();
        assert!(extracted.system.is_some());

        if let Some(ref sys_ref) = extracted.system {
            assert_eq!(extracted.get_text(sys_ref).unwrap(), "System instruction.");
        }
    }

    #[test]
    fn test_set_text_string_content() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "Old system.",
            "messages": [
                { "role": "user", "content": "Old user text." }
            ]
        });

        let mut extracted = extract_text_messages(body).unwrap();

        let user_ref = extracted.user[0].clone();
        extracted.set_text(&user_ref, "New user text.".to_string());

        let sys_ref = extracted.system.clone().unwrap();
        extracted.set_text(&sys_ref, "New system.".to_string());

        let final_body = extracted.into_body();
        assert_eq!(final_body["system"], "New system.");
        assert_eq!(final_body["messages"][0]["content"], "New user text.");
    }

    #[test]
    fn test_set_text_block_content() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Original text." },
                        { "type": "image", "source": { "type": "base64", "data": "..." } }
                    ]
                }
            ]
        });

        let mut extracted = extract_text_messages(body).unwrap();
        let user_ref = extracted.user[0].clone();
        extracted.set_text(&user_ref, "Modified text.".to_string());

        let final_body = extracted.into_body();
        let content = &final_body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Modified text.");
        assert_eq!(content[1]["type"], "image");
    }

    #[test]
    fn test_no_system_field() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                { "role": "user", "content": "Hello" }
            ]
        });

        let extracted = extract_text_messages(body).unwrap();
        assert!(extracted.system.is_none());
        assert!(!extracted.has_system());
    }

    #[test]
    fn test_skip_messages_without_text() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "image", "source": { "type": "base64", "data": "..." } }
                    ]
                },
                {
                    "role": "user",
                    "content": "This has text."
                }
            ]
        });

        let extracted = extract_text_messages(body).unwrap();
        assert_eq!(extracted.user.len(), 1);
    }
}
