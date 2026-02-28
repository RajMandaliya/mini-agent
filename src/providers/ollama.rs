/// Ollama provider — runs models locally via http://localhost:11434.
/// Ollama exposes an OpenAI-compatible `/v1/chat/completions` endpoint
/// since v0.1.24, so we reuse the shared OpenAI helpers.
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

use crate::{AgentError, Completion, LlmProvider, Message, Tool};
use super::{build_openai_messages, build_openai_tools, parse_openai_completion};

pub struct OllamaProvider {
    client: Client,
    base_url: String,
    default_model: String,
}

impl OllamaProvider {
    /// Uses `http://localhost:11434` by default.
    /// `model` – any locally pulled Ollama model, e.g. `"llama3"`, `"mistral"`, `"qwen2"`.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base_url("http://localhost:11434", model)
    }

    /// Use a custom Ollama host (e.g. a remote server or Docker container).
    pub fn with_base_url(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            default_model: model.into(),
        }
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn provider_name(&self) -> &str { "Ollama" }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[&dyn Tool],
        model: &str,
    ) -> Result<Completion, AgentError> {
        let active_model = if model.is_empty() { &self.default_model } else { model };

        let msgs_json = build_openai_messages(messages);
        let tools_json = build_openai_tools(tools);

        let body = json!({
            "model": active_model,
            "messages": msgs_json,
            "tools": if tools_json.is_empty() { serde_json::Value::Null } else { json!(tools_json) },
            "stream": false,
        });

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentError::ProviderError(format!(
                    "Ollama unreachable at {} — is it running? ({})",
                    self.base_url, e
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::InvalidResponse(format!("Ollama {status}: {text}")));
        }

        let json: serde_json::Value = response.json().await?;
        parse_openai_completion(&json)
    }
}