/// OpenRouter provider — with streaming support.
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;

use super::{build_openai_messages, build_openai_tools, parse_openai_completion};
use crate::{parse_sse_chunk, AgentError, Completion, LlmProvider, Message, TokenStream, Tool};

pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
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
    fn provider_name(&self) -> &str {
        "OpenRouter"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[&dyn Tool],
        model: &str,
    ) -> Result<Completion, AgentError> {
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
            let message = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|j| j["error"]["message"].as_str().map(str::to_string))
                .unwrap_or(text);
            return Err(AgentError::provider("OpenRouter", message, Some(status)));
        }

        let json: serde_json::Value = response.json().await?;
        parse_openai_completion(&json)
    }

    async fn stream_complete(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<TokenStream, AgentError> {
        let active_model = if model.is_empty() { &self.model } else { model };
        let msgs_json = build_openai_messages(messages);

        let body = json!({
            "model": active_model,
            "messages": msgs_json,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": true,
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
            let message = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|j| j["error"]["message"].as_str().map(str::to_string))
                .unwrap_or(text);
            return Err(AgentError::provider("OpenRouter", message, Some(status)));
        }

        let byte_stream = response.bytes_stream();
        let token_stream = byte_stream
            .map(|r| r.map_err(AgentError::Network))
            .flat_map(|result| {
                let lines: Vec<Result<String, AgentError>> = match result {
                    Err(e) => vec![Err(e)],
                    Ok(bytes) => String::from_utf8_lossy(&bytes)
                        .split('\n')
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| Ok(l.to_string()))
                        .collect(),
                };
                futures::stream::iter(lines)
            })
            .filter_map(|line_result| async move {
                match line_result {
                    Err(e) => Some(Err(e)),
                    Ok(line) => parse_sse_chunk(&line).map(Ok),
                }
            });

        Ok(Box::pin(token_stream))
    }
}
