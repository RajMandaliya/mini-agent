/// Native OpenAI provider (api.openai.com).
/// Uses the same OpenAI-compatible message/tool shape as OpenRouter.
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

use crate::{AgentError, Completion, LlmProvider, Message, Tool};
use super::{build_openai_messages, build_openai_tools, parse_openai_completion};

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    default_model: String,
}

impl OpenAiProvider {
    /// `model` – e.g. `"gpt-4o"`, `"gpt-4o-mini"`, `"gpt-3.5-turbo"`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            default_model: model.into(),
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn provider_name(&self) -> &str { "OpenAI" }

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
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 1024,
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::InvalidResponse(format!("OpenAI {status}: {text}")));
        }

        let json: serde_json::Value = response.json().await?;
        parse_openai_completion(&json)
    }
}