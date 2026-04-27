/// Native OpenAI provider (api.openai.com) — with streaming support.
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;

use super::{build_openai_messages, build_openai_tools, parse_openai_completion};
use crate::{parse_sse_chunk, AgentError, Completion, LlmProvider, Message, TokenStream, Tool};

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
    fn provider_name(&self) -> &str {
        "OpenAI"
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
        let active_model = if model.is_empty() {
            &self.default_model
        } else {
            model
        };
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
            let message = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|j| j["error"]["message"].as_str().map(str::to_string))
                .unwrap_or(text);
            return Err(AgentError::provider("OpenAI", message, Some(status)));
        }

        let json: serde_json::Value = response.json().await?;
        parse_openai_completion(&json)
    }

    /// Streaming completion via OpenAI SSE.
    ///
    /// Sends `"stream": true` in the request body. The response is a sequence
    /// of server-sent events. Each line starting with `data: ` contains a JSON
    /// delta. We extract `choices[0].delta.content` from each event and yield
    /// it as a chunk. The stream ends when we receive `data: [DONE]`.
    async fn stream_complete(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<TokenStream, AgentError> {
        let active_model = if model.is_empty() {
            &self.default_model
        } else {
            model
        };
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
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
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
            return Err(AgentError::provider("OpenAI", message, Some(status)));
        }

        // Stream the response byte-by-byte, splitting on newlines.
        // Each complete line is parsed as an SSE event.
        let byte_stream = response.bytes_stream();

        let token_stream = byte_stream
            .map(|result| result.map_err(AgentError::Network))
            // Accumulate bytes into lines
            .flat_map(|result| {
                let lines: Vec<Result<String, AgentError>> = match result {
                    Err(e) => vec![Err(e)],
                    Ok(bytes) => String::from_utf8_lossy(&bytes)
                        .split('\n')
                        .filter(|line| !line.trim().is_empty())
                        .map(|line| Ok(line.to_string()))
                        .collect(),
                };
                futures::stream::iter(lines)
            })
            // Parse each SSE line into a text chunk
            .filter_map(|line_result| async move {
                match line_result {
                    Err(e) => Some(Err(e)),
                    Ok(line) => parse_sse_chunk(&line).map(Ok),
                }
            });

        Ok(Box::pin(token_stream))
    }
}
