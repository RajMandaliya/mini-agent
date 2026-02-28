/// OpenRouter provider — original provider, now wired to the shared LlmProvider trait.
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

use crate::{AgentError, Completion, LlmProvider, Message, Tool};
use super::{build_openai_messages, build_openai_tools, parse_openai_completion};

pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
    /// `model` – any OpenRouter model slug, e.g. `"meta-llama/llama-3.1-8b-instruct"`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    fn provider_name(&self) -> &str { "OpenRouter" }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[&dyn Tool],
        model: &str,
    ) -> Result<Completion, AgentError> {
        // Use per-call model override if provided, else fall back to default
        let active_model = if model.is_empty() { &self.model } else { model };

        let msgs_json = build_openai_messages(messages);
        let tools_json = build_openai_tools(tools);

        let body = json!({
            "model": active_model,
            "messages": msgs_json,
            "tools": if tools_json.is_empty() { serde_json::Value::Null } else { json!(tools_json) },
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 1024,
        });

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://github.com/RajMandaliya/mini-agent")
            .header("X-Title", "mini-agent")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::InvalidResponse(format!("OpenRouter {status}: {text}")));
        }

        let json: serde_json::Value = response.json().await?;
        parse_openai_completion(&json)
    }
}