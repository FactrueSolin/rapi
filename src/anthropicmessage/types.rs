use crate::anthropicmessage::{ExtractedMessages, MessageRef, MessageSource};

pub struct AnthropicMessageEntry {
    index: usize,
    source: MessageSource,
    role: String,
    text: String,
}

impl AnthropicMessageEntry {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn source(&self) -> &MessageSource {
        &self.source
    }

    pub fn role(&self) -> &str {
        &self.role
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

pub struct AnthropicPluginMessageView {
    entries: Vec<AnthropicMessageEntry>,
}

impl AnthropicPluginMessageView {
    pub fn from_extracted(extracted: &ExtractedMessages) -> Self {
        let mut entries = Vec::new();

        if let Some(ref sys_ref) = extracted.system {
            if let Some(text) = extracted.get_text(sys_ref) {
                entries.push(AnthropicMessageEntry {
                    index: 0,
                    source: MessageSource::System,
                    role: "system".to_string(),
                    text,
                });
            }
        }

        let mut msg_index = 0;

        let all_refs: Vec<(&[MessageRef], &str)> = vec![
            (&extracted.user, "user"),
            (&extracted.assistant, "assistant"),
        ];

        for (refs, role) in all_refs {
            for msg_ref in refs {
                if let Some(text) = extracted.get_text(msg_ref) {
                    if let MessageSource::Messages(idx) = msg_ref.source() {
                        msg_index = *idx;
                    }

                    entries.push(AnthropicMessageEntry {
                        index: msg_index,
                        source: msg_ref.source().clone(),
                        role: role.to_string(),
                        text,
                    });
                }
            }
        }

        entries.sort_by_key(|e| match e.source {
            MessageSource::System => 0,
            MessageSource::Messages(idx) => idx + 1,
        });

        Self { entries }
    }

    pub fn entries(&self) -> &[AnthropicMessageEntry] {
        &self.entries
    }

    pub fn get_text_by_source(&self, source: &MessageSource) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| &e.source == source)
            .map(|e| e.text.as_str())
    }

    pub fn get_role_by_source(&self, source: &MessageSource) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| &e.source == source)
            .map(|e| e.role.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropicmessage::extract_text_messages;

    #[test]
    fn test_view_ordering() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "System prompt.",
            "messages": [
                { "role": "user", "content": "Hello" },
                { "role": "assistant", "content": "Hi" },
                { "role": "user", "content": "Bye" }
            ]
        });

        let extracted = extract_text_messages(body).unwrap();
        let view = AnthropicPluginMessageView::from_extracted(&extracted);

        let entries = view.entries();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].role(), "system");
        assert_eq!(entries[1].role(), "user");
        assert_eq!(entries[2].role(), "assistant");
        assert_eq!(entries[3].role(), "user");
    }

    #[test]
    fn test_view_without_system() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                { "role": "user", "content": "Hello" }
            ]
        });

        let extracted = extract_text_messages(body).unwrap();
        let view = AnthropicPluginMessageView::from_extracted(&extracted);

        let entries = view.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].role(), "user");
    }
}
