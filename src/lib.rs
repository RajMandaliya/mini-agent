pub mod providers;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fmt;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

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

    #[error("Provider error: {0}")]
    ProviderError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Message / Role
// ─────────────────────────────────────────────────────────────────────────────

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
        Self { role: Role::User, content: content.into(), tool_call_id: None, tool_calls: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), tool_call_id: None, tool_calls: None }
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

// ─────────────────────────────────────────────────────────────────────────────
// ToolCall / Completion
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Tool trait
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters_schema(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<String, AgentError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// LlmProvider trait
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn provider_name(&self) -> &str;

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[&dyn Tool],
        model: &str,
    ) -> Result<Completion, AgentError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export built-in providers
// ─────────────────────────────────────────────────────────────────────────────

pub use providers::openrouter::OpenRouterProvider;
pub use providers::openai::OpenAiProvider;
pub use providers::anthropic::AnthropicProvider;
pub use providers::ollama::OllamaProvider;

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Tools
// ─────────────────────────────────────────────────────────────────────────────

pub struct AddNumbersTool;

#[async_trait]
impl Tool for AddNumbersTool {
    fn name(&self) -> &'static str { "add_numbers" }
    fn description(&self) -> &'static str { "Adds two integers and returns the result" }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"],
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
    fn description(&self) -> &'static str { "Multiplies two integers and returns the result" }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"],
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
    fn description(&self) -> &'static str { "Fetches a random family-friendly joke and returns it" }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {}, "additionalProperties": false })
    }
    async fn execute(&self, _args: Value) -> Result<String, AgentError> {
        // Blacklist offensive categories
        let url = "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist,explicit,religious,political";
        let body = reqwest::get(url).await?.text().await?;
        let json: Value = serde_json::from_str(&body)?;
        let joke = if json["type"] == "single" {
            json["joke"].as_str().unwrap_or("No joke found").to_string()
        } else {
            format!(
                "{} {}",
                json["setup"].as_str().unwrap_or(""),
                json["delivery"].as_str().unwrap_or("")
            )
        };
        Ok(joke)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent
// ─────────────────────────────────────────────────────────────────────────────

pub struct Agent {
    pub provider: Box<dyn LlmProvider>,
    pub model: String,
    pub tools: Vec<Box<dyn Tool>>,
    pub history: Vec<Message>,
    pub max_steps: usize,
    pub system_prompt: String,
}

impl Agent {
    pub fn new(provider: Box<dyn LlmProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            tools: vec![],
            history: vec![],
            max_steps: 6,
            system_prompt: "You are a helpful assistant. Only call tools that are directly needed to answer the question. Never call unrelated tools. Once you receive a tool result, use it to give the final answer immediately.".to_string(),
        }
    }

    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.push(Box::new(tool));
    }

    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError> {
        self.history.push(Message::user(user_input));
        let mut executed_tool_calls = HashSet::new();

        for step in 0..self.max_steps {
            let tool_refs: Vec<&dyn Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

            // Inject system prompt as first message on every call
            let mut messages = vec![Message {
                role: Role::User,
                content: format!("[SYSTEM]: {}", self.system_prompt),
                tool_call_id: None,
                tool_calls: None,
            }];
            messages.extend(self.history.clone());

            let completion = self
                .provider
                .complete(&messages, &tool_refs, &self.model)
                .await
                .map_err(|e| {
                    AgentError::ProviderError(format!(
                        "[{}] step {}: {}",
                        self.provider.provider_name(),
                        step,
                        e
                    ))
                })?;

            let content = completion.content.clone().unwrap_or_default();
            let tool_calls = completion.tool_calls.clone();
            let raw_tool_calls = completion.raw_tool_calls.clone().unwrap_or(Value::Null);

            self.history.push(Message::assistant_with_tools(
                content.clone(),
                raw_tool_calls,
            ));

            // No tool calls — final answer
            if tool_calls.is_empty() {
                if !content.is_empty() {
                    return Ok(content);
                }
                return Err(AgentError::ProviderError("Empty response from model".to_string()));
            }

            // Execute tools
            let mut executed_any = false;
            for call in &tool_calls {
                if executed_tool_calls.contains(&call.id) {
                    continue;
                }

                println!(
                    "[{}] Executing tool: {}",
                    self.provider.provider_name(),
                    call.name
                );
                let result = self.execute_tool(call).await?;
                executed_tool_calls.insert(call.id.clone());

                self.history.push(Message {
                    role: Role::Tool,
                    content: result,
                    tool_call_id: Some(call.id.clone()),
                    tool_calls: None,
                });

                executed_any = true;
            }

            if !executed_any {
                // All were duplicates
                if !content.is_empty() {
                    return Ok(content);
                }
                return Err(AgentError::ProviderError(
                    "Duplicate tool calls with no content".to_string(),
                ));
            }

            // Loop back to get model's response to tool results
        }

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