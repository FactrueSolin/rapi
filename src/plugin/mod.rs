pub mod openai_privacy;
pub mod types;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use crate::anthropicmessage::{ExtractedMessages as AnthropicExtractedMessages, MessageRef as AnthropicMessageRef, MessageSource};
use crate::openaichatcompletion::{ExtractedMessages, MessageRef};

use self::types::{PluginMessageView, PluginResult, Replacement};

#[async_trait]
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, messages: &PluginMessageView) -> Result<PluginResult, String>;
}

pub struct PluginRegistry {
    plugins: Vec<Arc<dyn Plugin>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    pub fn register(&mut self, plugin: Arc<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    pub async fn process_messages(
        &self,
        extracted: ExtractedMessages,
    ) -> Result<ExtractedMessages, String> {
        let view = PluginMessageView::from_extracted(&extracted);

        let futures: Vec<_> = self
            .plugins
            .iter()
            .map(|plugin| {
                let plugin = plugin.clone();
                let view = &view;
                async move {
                    let result = plugin.process(view).await;
                    (plugin.name().to_string(), result)
                }
            })
            .collect();

        let results = join_all(futures).await;

        let mut replacements_by_index: HashMap<usize, Vec<Replacement>> = HashMap::new();

        for (plugin_name, result) in results {
            match result {
                Ok(plugin_result) => {
                    for msg_repl in plugin_result.message_replacements {
                        replacements_by_index
                            .entry(msg_repl.message_index)
                            .or_default()
                            .extend(msg_repl.replacements);
                    }
                }
                Err(e) => {
                    eprintln!("Plugin '{}' error: {}", plugin_name, e);
                }
            }
        }

        let mut extracted = extracted;
        for (index, replacements) in replacements_by_index {
            let msg_ref = MessageRef::new(index);
            if let Some(current_text) = extracted.get_text(&msg_ref) {
                let new_text = merge_replacements(&current_text, replacements);
                extracted.set_text(&msg_ref, new_text);
            }
        }

        Ok(extracted)
    }

    pub async fn process_anthropic_messages(
        &self,
        extracted: AnthropicExtractedMessages,
        extracted_view: &PluginMessageView,
    ) -> Result<AnthropicExtractedMessages, String> {
        const SYSTEM_INDEX: usize = usize::MAX;

        let futures: Vec<_> = self
            .plugins
            .iter()
            .map(|plugin| {
                let plugin = plugin.clone();
                let view = &extracted_view;
                async move {
                    let result = plugin.process(view).await;
                    (plugin.name().to_string(), result)
                }
            })
            .collect();

        let results = join_all(futures).await;

        let mut replacements_by_source: HashMap<MessageSource, Vec<Replacement>> = HashMap::new();

        for (plugin_name, result) in results {
            match result {
                Ok(plugin_result) => {
                    for msg_repl in plugin_result.message_replacements {
                        let source = if msg_repl.message_index == SYSTEM_INDEX {
                            MessageSource::System
                        } else {
                            MessageSource::Messages(msg_repl.message_index)
                        };
                        replacements_by_source
                            .entry(source)
                            .or_default()
                            .extend(msg_repl.replacements);
                    }
                }
                Err(e) => {
                    eprintln!("Plugin '{}' error: {}", plugin_name, e);
                }
            }
        }

        let mut extracted = extracted;
        for (source, replacements) in replacements_by_source {
            let msg_ref = AnthropicMessageRef::from_source(source);
            if let Some(current_text) = extracted.get_text(&msg_ref) {
                let new_text = merge_replacements(&current_text, replacements);
                extracted.set_text(&msg_ref, new_text);
            }
        }

        Ok(extracted)
    }
}

fn merge_replacements(text: &str, replacements: Vec<Replacement>) -> String {
    if replacements.is_empty() {
        return text.to_string();
    }

    struct Accepted {
        start: usize,
        end: usize,
        replacement: String,
    }

    let mut accepted: Vec<Accepted> = Vec::new();

    for repl in replacements {
        if let Some(start) = text.find(&repl.original) {
            let end = start + repl.original.len();

            let overlap_indices: Vec<usize> = accepted
                .iter()
                .enumerate()
                .filter(|(_, a)| start < a.end && end > a.start)
                .map(|(i, _)| i)
                .collect();

            for i in overlap_indices.into_iter().rev() {
                accepted.remove(i);
            }

            accepted.push(Accepted {
                start,
                end,
                replacement: repl.replacement,
            });
        }
    }

    accepted.sort_by_key(|a| a.start);

    let mut result = text.to_string();
    for accepted in accepted.into_iter().rev() {
        result.replace_range(accepted.start..accepted.end, &accepted.replacement);
    }

    result
}
