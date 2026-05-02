use serde_json::Value;

pub struct MessageRef {
    index: usize,
}

impl MessageRef {
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

pub struct ExtractedMessages {
    pub system: Vec<MessageRef>,
    pub developer: Vec<MessageRef>,
    pub assistant: Vec<MessageRef>,
    pub user: Vec<MessageRef>,
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

pub fn extract_text_messages(body: Value) -> Result<ExtractedMessages, String> {
    let messages: Vec<(usize, Value)> = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Missing or invalid 'messages' field".to_string())?
        .iter()
        .enumerate()
        .map(|(i, msg)| (i, msg.clone()))
        .collect();

    let mut result = ExtractedMessages {
        system: Vec::new(),
        developer: Vec::new(),
        assistant: Vec::new(),
        user: Vec::new(),
        body,
    };

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
            let msg_ref = MessageRef { index: i };
            match role {
                "system" => result.system.push(msg_ref),
                "developer" => result.developer.push(msg_ref),
                "assistant" => result.assistant.push(msg_ref),
                "user" => result.user.push(msg_ref),
                _ => {}
            }
        }
    }

    Ok(result)
}

impl ExtractedMessages {
    pub fn get_text(&self, msg: &MessageRef) -> Option<String> {
        let content = self
            .body
            .get("messages")
            .and_then(Value::as_array)?
            .get(msg.index)
            .and_then(|m| m.get("content"))?;

        if content.is_null() {
            return None;
        }

        get_text_from_content(content)
    }

    pub fn set_text(&mut self, msg: &MessageRef, text: String) {
        if let Some(messages) = self.body.get_mut("messages").and_then(Value::as_array_mut) {
            if let Some(message) = messages.get_mut(msg.index) {
                if let Some(content) = message.get_mut("content") {
                    set_text_to_content(content, text);
                }
            }
        }
    }

    pub fn into_body(self) -> Value {
        self.body
    }
}
