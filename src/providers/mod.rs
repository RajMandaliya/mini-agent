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

pub(crate) fn build_openai_messages(messages: &[Message]) -> Vec<Value> {
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

pub(crate) fn build_openai_tools(tools: &[&dyn Tool]) -> Vec<Value> {
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

pub(crate) fn parse_openai_completion(json: &Value) -> Result<Completion, AgentError> {
    let choice = json
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| AgentError::InvalidResponse("missing 'choices'".into()))?;

    let message = choice
        .get("message")
        .ok_or_else(|| AgentError::InvalidResponse("missing 'message'".into()))?;

    let content = message.get("content").and_then(|v| v.as_str()).map(str::to_string);
    let raw_tool_calls = message.get("tool_calls").cloned();

    let mut tool_calls: Vec<ToolCall> = vec![];
    if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for call in calls {
            let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let function = call.get("function").ok_or_else(|| {
                AgentError::InvalidResponse("missing function in tool call".into())
            })?;
            let name =
                function.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let args_raw = function
                .get("arguments")
                .ok_or_else(|| AgentError::InvalidResponse("missing arguments".into()))?;
            let args: Value = if let Some(s) = args_raw.as_str() {
                serde_json::from_str(s)
                    .map_err(|e| AgentError::InvalidResponse(format!("bad args JSON: {e}")))?
            } else {
                args_raw.clone()
            };
            tool_calls.push(ToolCall { id, name, args });
        }
    }

    Ok(Completion { content, tool_calls, raw_tool_calls })
}