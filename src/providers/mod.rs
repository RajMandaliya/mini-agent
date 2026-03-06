pub mod anthropic;
pub mod ollama;
pub mod openai;
pub mod openrouter;

use crate::{AgentError, Completion, Message, Tool, ToolCall};
use serde_json::Value;

// ─────────────────────────────────────────────────────────────────────────────
// Shared OpenAI-compatible helpers
// (used by OpenRouter + OpenAI — they share the same API shape)
// ─────────────────────────────────────────────────────────────────────────────

pub fn build_openai_messages(messages: &[Message]) -> Vec<Value> {
    use serde_json::json;
    messages
        .iter()
        .map(|m| {
            let mut obj = json!({
                "role": m.role,
                "content": m.content,
            });
            if let Some(id) = &m.tool_call_id {
                obj["tool_call_id"] = json!(id);
            }
            if let Some(tc) = &m.tool_calls {
                if !tc.is_null() {
                    obj["tool_calls"] = tc.clone();
                }
            }
            obj
        })
        .collect()
}

pub fn build_openai_tools(tools: &[&dyn Tool]) -> Vec<Value> {
    use serde_json::json;
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.parameters_schema(),
                }
            })
        })
        .collect()
}

pub fn parse_openai_completion(json: &Value) -> Result<Completion, AgentError> {
    // Provider name isn't known here, so we use a placeholder; callers may
    // wrap this error to add more context if desired.
    let provider = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown provider")
        .to_string();

    let choice = json
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| AgentError::invalid(&provider, "response missing 'choices' array — the API may have returned an error body"))?;

    let message = choice
        .get("message")
        .ok_or_else(|| AgentError::invalid(&provider, "choice is missing 'message' field"))?;

    let content = message.get("content").and_then(|v| v.as_str()).map(str::to_string);
    let raw_tool_calls = message.get("tool_calls").cloned();

    let mut tool_calls: Vec<ToolCall> = vec![];
    if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for (i, call) in calls.iter().enumerate() {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let function = call.get("function").ok_or_else(|| {
                AgentError::invalid(&provider, format!("tool_call[{i}] is missing 'function' field"))
            })?;
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args_raw = function.get("arguments").ok_or_else(|| {
                AgentError::invalid(&provider, format!("tool_call[{i}] function '{name}' is missing 'arguments'"))
            })?;
            let args: Value = if let Some(s) = args_raw.as_str() {
                serde_json::from_str(s).map_err(|e| {
                    AgentError::invalid(
                        &provider,
                        format!("tool_call[{i}] '{name}' has invalid JSON arguments: {e}"),
                    )
                })?
            } else {
                args_raw.clone()
            };
            tool_calls.push(ToolCall { id, name, args });
        }
    }

    Ok(Completion { content, tool_calls, raw_tool_calls })
}