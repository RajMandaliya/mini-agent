use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("LLM request failed: {0}")]
    LlmError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid response from LLM: {0}")]
    InvalidResponse(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution failed: {0}")]
    ToolError(String),

    #[error("Max iterations reached")]
    MaxIterations,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            }
        )
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Value) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Debug)]
pub struct Completion {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub raw_tool_calls: Option<Value>,
}

#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters_schema(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<String, AgentError>;
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[&dyn Tool],
        model: &str,
    ) -> Result<Completion, AgentError>;
}

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
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[&dyn Tool],
        _model_override: &str,
    ) -> Result<Completion, AgentError> {
        const URL: &str = "https://openrouter.ai/api/v1/chat/completions";

        let msgs_json: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut obj = json!({
                    "role": m.role,
                    "content": m.content,
                });
                if let Some(id) = &m.tool_call_id {
                    obj["tool_call_id"] = json!(id);
                }
                obj
            })
            .collect();

        let tools_json: Vec<Value> = tools
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
            .collect();

        let body = json!({
            "model": self.model,
            "messages": msgs_json,
            "tools": if tools_json.is_empty() { Value::Null } else { json!(tools_json) },
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 1024,
        });

        let response = self
            .client
            .post(URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "https://github.com/YOUR_USERNAME/mini-agent")
            .header("X-Title", "mini-agent Rust demo")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(AgentError::InvalidResponse(format!(
                "OpenRouter returned {}: {}",
                status, body_text
            )));
        }

        let json: Value = response.json().await?;
        let choice = json
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| AgentError::InvalidResponse("missing 'choices'".to_string()))?;

        let message = choice
            .get("message")
            .ok_or_else(|| AgentError::InvalidResponse("missing 'message'".to_string()))?;

        let content = message.get("content").and_then(|v| v.as_str()).map(str::to_string);

        let mut tool_calls = Vec::new();
        let raw_tool_calls = message.get("tool_calls").cloned();

        if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for call in calls {
                let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let function = call.get("function").ok_or_else(|| {
                    AgentError::InvalidResponse("missing function in tool call".to_string())
                })?;
                let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let args_value = function.get("arguments").ok_or_else(|| {
                    AgentError::InvalidResponse("missing arguments".to_string())
                })?;
                let args: Value = if let Some(s) = args_value.as_str() {
                    serde_json::from_str(s).map_err(|e| {
                        AgentError::InvalidResponse(format!("invalid tool args string: {}", e))
                    })?
                } else {
                    args_value.clone()
                };
                tool_calls.push(ToolCall { id, name, args });
            }
        }

        Ok(Completion {
            content,
            tool_calls,
            raw_tool_calls,
        })
    }
}

/// -------------------- TOOLS --------------------

pub struct AddNumbersTool;

#[async_trait]
impl Tool for AddNumbersTool {
    fn name(&self) -> &'static str { "add_numbers" }
    fn description(&self) -> &'static str { "Adds two integers and returns the result as a string" }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a","b"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<String, AgentError> {
        let a = args["a"].as_i64().ok_or_else(|| AgentError::ToolError("Missing 'a'".into()))?;
        let b = args["b"].as_i64().ok_or_else(|| AgentError::ToolError("Missing 'b'".into()))?;
        Ok((a + b).to_string())
    }
}

pub struct MultiplyNumbersTool;

#[async_trait]
impl Tool for MultiplyNumbersTool {
    fn name(&self) -> &'static str { "multiply_numbers" }
    fn description(&self) -> &'static str { "Multiplies two integers and returns the result as a string" }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a","b"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<String, AgentError> {
        let a = args["a"].as_i64().ok_or_else(|| AgentError::ToolError("Missing 'a'".into()))?;
        let b = args["b"].as_i64().ok_or_else(|| AgentError::ToolError("Missing 'b'".into()))?;
        Ok((a * b).to_string())
    }
}

pub struct JokeTool;

#[async_trait]
impl Tool for JokeTool {
    fn name(&self) -> &'static str { "get_joke" }
    fn description(&self) -> &'static str { "Fetches a joke from JokeAPI and returns the text" }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value) -> Result<String, AgentError> {
        let url = "https://v2.jokeapi.dev/joke/Any";
        let body = reqwest::get(url).await?.text().await?;
        let json: Value = serde_json::from_str(&body)?;
        let joke = if json["type"] == "single" {
            json["joke"].as_str().unwrap_or("").to_string()
        } else {
            format!("{} {}", json["setup"].as_str().unwrap_or(""), json["delivery"].as_str().unwrap_or(""))
        };
        Ok(joke)
    }
}

/// -------------------- AGENT --------------------

pub struct Agent {
    pub provider: Box<dyn LlmProvider>,
    pub model: String,
    pub tools: Vec<Box<dyn Tool>>,
    pub history: Vec<Message>,
    pub max_steps: usize,
}

impl Agent {
    pub fn new(provider: Box<dyn LlmProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            tools: vec![],
            history: vec![],
            max_steps: 6,
        }
    }

    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.push(Box::new(tool));
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError> {
    self.history.push(Message::user(user_input));
    let mut executed_tool_calls = HashSet::new();

    for _ in 0..self.max_steps {
        // Prepare tool references
        let tool_refs: Vec<&dyn Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

        // Get completion from LLM
        let completion = self
            .provider
            .complete(&self.history, &tool_refs, &self.model)
            .await?;

        // Push assistant message (with tool_calls if present)
        self.history.push(Message::assistant_with_tools(
            completion.content.clone().unwrap_or_default(),
            completion.raw_tool_calls.clone().unwrap_or(Value::Null),
        ));

        // Execute any new tool calls
        if !completion.tool_calls.is_empty() {
            let mut executed_any = false;

            for call in &completion.tool_calls {
                if executed_tool_calls.contains(&call.id) {
                    continue; // skip already executed
                }

                println!("Executing tool: {}", call.name);
                let result = self.execute_tool(call).await?;
                executed_tool_calls.insert(call.id.clone());

                // Push tool result to history
                self.history.push(Message {
                    role: Role::Tool,
                    content: result.clone(),
                    tool_call_id: Some(call.id.clone()),
                    tool_calls: None,
                });

                executed_any = true;

                // Return first executed tool result immediately
                return Ok(result);
            }

            // If any new tool was executed, loop again to let LLM see results
            if executed_any {
                continue;
            }
        }

        // If LLM returned content without tools, return it
        if let Some(content) = completion.content {
            return Ok(content);
        }
    }

    // Max iterations reached
    Err(AgentError::MaxIterations)
}
    async fn execute_tool(&self, call: &ToolCall) -> Result<String, AgentError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == call.name)
            .ok_or_else(|| AgentError::ToolNotFound(call.name.clone()))?;
        tool.execute(call.args.clone()).await
    }
}