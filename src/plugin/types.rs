use crate::openaichatcompletion::{ExtractedMessages, MessageRef};

pub struct Replacement {
    pub original: String,
    pub replacement: String,
}

pub struct MessageReplacements {
    pub message_index: usize,
    pub replacements: Vec<Replacement>,
}

pub struct PluginResult {
    pub message_replacements: Vec<MessageReplacements>,
}

pub struct MessageEntry {
    index: usize,
    role: String,
    text: String,
}

impl MessageEntry {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn role(&self) -> &str {
        &self.role
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

pub struct PluginMessageView {
    entries: Vec<MessageEntry>,
}

impl PluginMessageView {
    pub fn from_extracted(extracted: &ExtractedMessages) -> Self {
        let mut entries = Vec::new();

        let all_refs: Vec<(&[MessageRef], &str)> = vec![
            (&extracted.system, "system"),
            (&extracted.developer, "developer"),
            (&extracted.user, "user"),
            (&extracted.assistant, "assistant"),
        ];

        for (refs, role) in all_refs {
            for msg_ref in refs {
                if let Some(text) = extracted.get_text(msg_ref) {
                    entries.push(MessageEntry {
                        index: msg_ref.index(),
                        role: role.to_string(),
                        text,
                    });
                }
            }
        }

        entries.sort_by_key(|e| e.index);

        Self { entries }
    }

    pub fn entries(&self) -> &[MessageEntry] {
        &self.entries
    }

    pub fn get_text(&self, index: usize) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.index == index)
            .map(|e| e.text.as_str())
    }

    pub fn get_role(&self, index: usize) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.index == index)
            .map(|e| e.role.as_str())
    }
}
