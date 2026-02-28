/// Anthropic provider — supports Claude models via api.anthropic.com.
/// Anthropic uses a different API shape (no tool_calls in the OpenAI sense),
/// so this provider translates to/from Anthropic's native format.
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use crate::{AgentError, Completion, LlmProvider, Message, Role, Tool, ToolCall};

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    default_model: String,
}

impl AnthropicProvider {
    /// `model` – e.g. `"claude-sonnet-4-20250514"`, `"claude-3-haiku-20240307"`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            default_model: model.into(),
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn provider_name(&self) -> &str { "Anthropic" }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[&dyn Tool],
        model: &str,
    ) -> Result<Completion, AgentError> {
        let active_model = if model.is_empty() { &self.default_model } else { model };

        // ── Convert messages ───────────────────────────────────────────────
        // Anthropic separates system messages and uses a content-block format
        // for tool results. We extract an optional leading system message.
        let mut system_prompt: Option<String> = None;
        let mut anthropic_messages: Vec<Value> = vec![];

        for msg in messages {
            match msg.role {
                Role::User => {
                    // A tool result coming back from the agent sits in a "user"
                    // turn in Anthropic's API as a tool_result content block.
                    if let Some(id) = &msg.tool_call_id {
                        anthropic_messages.push(json!({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": msg.content,
                            }]
                        }));
                    } else {
                        anthropic_messages.push(json!({
                            "role": "user",
                            "content": msg.content,
                        }));
                    }
                }
                Role::Tool => {
                    // In our agent the tool result is stored with Role::Tool.
                    // Anthropic expects it as a user-turn tool_result block.
                    if let Some(id) = &msg.tool_call_id {
                        anthropic_messages.push(json!({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": msg.content,
                            }]
                        }));
                    }
                }
                Role::Assistant => {
                    // If the assistant turn contained tool_calls we need to
                    // re-emit them as tool_use blocks so Anthropic recognises
                    // the assistant turn correctly.
                    if let Some(tc) = &msg.tool_calls {
                        if let Some(calls) = tc.as_array() {
                            let content_blocks: Vec<Value> = calls
                                .iter()
                                .filter_map(|c| {
                                    let id = c.get("id")?.as_str()?;
                                    let name =
                                        c.get("function")?.get("name")?.as_str()?;
                                    let raw_args =
                                        c.get("function")?.get("arguments")?;
                                    let input: Value =
                                        if let Some(s) = raw_args.as_str() {
                                            serde_json::from_str(s).ok()?
                                        } else {
                                            raw_args.clone()
                                        };
                                    Some(json!({
                                        "type": "tool_use",
                                        "id": id,
                                        "name": name,
                                        "input": input,
                                    }))
                                })
                                .collect();

                            let mut blocks = vec![];
                            if !msg.content.is_empty() {
                                blocks.push(json!({ "type": "text", "text": msg.content }));
                            }
                            blocks.extend(content_blocks);

                            anthropic_messages
                                .push(json!({ "role": "assistant", "content": blocks }));
                            continue;
                        }
                    }
                    anthropic_messages.push(json!({
                        "role": "assistant",
                        "content": msg.content,
                    }));
                }
            }
        }

        // ── Build tools ────────────────────────────────────────────────────
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "description": t.description(),
                    "input_schema": t.parameters_schema(),
                })
            })
            .collect();

        // ── Request body ───────────────────────────────────────────────────
        let mut body = json!({
            "model": active_model,
            "max_tokens": 1024,
            "messages": anthropic_messages,
        });

        if let Some(sys) = system_prompt {
            body["system"] = json!(sys);
        }

        if !anthropic_tools.is_empty() {
            body["tools"] = json!(anthropic_tools);
        }

        // ── HTTP call ──────────────────────────────────────────────────────
        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::InvalidResponse(format!("Anthropic {status}: {text}")));
        }

        let json: Value = response.json().await?;

        // ── Parse response ─────────────────────────────────────────────────
        let content_blocks = json
            .get("content")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AgentError::InvalidResponse("missing 'content' array".into()))?;

        let mut text_parts: Vec<String> = vec![];
        let mut tool_calls: Vec<ToolCall> = vec![];
        // We also build a raw_tool_calls Value in OpenAI shape so the agent
        // history stores something consistent.
        let mut raw_tool_calls_arr: Vec<Value> = vec![];

        for block in content_blocks {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(t.to_string());
                    }
                }
                Some("tool_use") => {
                    let id =
                        block.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let name =
                        block.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let args = block.get("input").cloned().unwrap_or(json!({}));

                    // Build OpenAI-compatible raw representation for history
                    raw_tool_calls_arr.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args.to_string(),
                        }
                    }));

                    tool_calls.push(ToolCall { id, name, args });
                }
                _ => {}
            }
        }

        let content = if text_parts.is_empty() { None } else { Some(text_parts.join("\n")) };
        let raw_tool_calls = if raw_tool_calls_arr.is_empty() {
            None
        } else {
            Some(json!(raw_tool_calls_arr))
        };

        Ok(Completion { content, tool_calls, raw_tool_calls })
    }
}