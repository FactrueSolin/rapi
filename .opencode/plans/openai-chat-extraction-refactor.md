# OpenAI Chat Completion 消息提取重构计划

## 文件变更

### 1. `src/openaichatcompletion/mod.rs` (完全重写)

```rust
use serde_json::Value;

pub struct MessageRef {
    index: usize,
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
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Missing or invalid 'messages' field".to_string())?;

    let mut result = ExtractedMessages {
        system: Vec::new(),
        developer: Vec::new(),
        assistant: Vec::new(),
        user: Vec::new(),
        body,
    };

    for (i, msg) in messages.iter().enumerate() {
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
```

### 2. `src/interceptor.rs` (更新调用方式)

```rust
use crate::openaichatcompletion;

pub fn should_intercept(path: &str) -> bool {
    path.ends_with("/chat/completions")
}

pub fn intercept_body(body: &[u8]) -> Result<Vec<u8>, String> {
    let json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| format!("Failed to parse request body as JSON: {}", e))?;

    eprintln!("=== INTERCEPTED REQUEST BODY ===");
    eprintln!("{}", serde_json::to_string_pretty(&json).unwrap_or_else(|_| "Failed to pretty print".to_string()));
    eprintln!("================================");

    let mut extracted = openaichatcompletion::extract_text_messages(json)?;

    eprintln!("=== EXTRACTED MESSAGES ===");
    for msg_ref in &extracted.system {
        if let Some(text) = extracted.get_text(msg_ref) {
            eprintln!("system: {}", text);
        }
    }
    for msg_ref in &extracted.developer {
        if let Some(text) = extracted.get_text(msg_ref) {
            eprintln!("developer: {}", text);
        }
    }
    for msg_ref in &extracted.assistant {
        if let Some(text) = extracted.get_text(msg_ref) {
            eprintln!("assistant: {}", text);
        }
    }
    for msg_ref in &extracted.user {
        if let Some(text) = extracted.get_text(msg_ref) {
            eprintln!("user: {}", text);
        }
    }
    eprintln!("==========================");

    // TODO: 在这里修改消息
    // 示例: 遍历修改所有 user 消息
    // for msg_ref in &extracted.user {
    //     if let Some(text) = extracted.get_text(msg_ref) {
    //         let modified = process_text(&text);
    //         extracted.set_text(msg_ref, modified);
    //     }
    // }

    let modified_body = serde_json::to_vec(&extracted.into_body())
        .map_err(|e| format!("Failed to serialize modified body: {}", e))?;

    Ok(modified_body)
}
```

## API 使用示例

```rust
// 提取消息
let body: serde_json::Value = /* 原始请求体 */;
let mut extracted = extract_text_messages(body)?;

// 获取文本
for msg_ref in &extracted.user {
    if let Some(text) = extracted.get_text(msg_ref) {
        println!("User message: {}", text);
    }
}

// 修改文本
for msg_ref in &extracted.user {
    if let Some(text) = extracted.get_text(msg_ref) {
        let modified = text.replace("sensitive", "[REDACTED]");
        extracted.set_text(msg_ref, modified);
    }
}

// 导出修改后的完整请求体
let new_body = extracted.into_body();
```

## 关键设计点

1. **MessageRef** 只暴露索引，不暴露内部实现细节
2. **ExtractedMessages** 持有原始 body 的所有权，确保数据一致性
3. **set_text** 智能处理 content 格式：
   - 原始是字符串 → 替换为字符串
   - 原始是数组 → 更新第一个 text 部分，若无则追加
   - 原始是 null → 创建字符串
4. **into_body** 消费自身，返回修改后的完整请求体，保留所有原始字段
