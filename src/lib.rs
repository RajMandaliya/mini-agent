pub mod providers;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Stream type alias
// ─────────────────────────────────────────────────────────────────────────────

/// A pinned, boxed stream of text chunks from the LLM.
/// Each item is one token or word as it arrives from the provider.
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String, AgentError>> + Send>>;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid response from {provider}: {message}")]
    InvalidResponse { provider: String, message: String },

    #[error("Tool not found: '{0}' — did you forget to call agent.add_tool(…)?")]
    ToolNotFound(String),

    #[error("Tool '{tool}' failed: {reason}")]
    ToolExecution { tool: String, reason: String },

    #[error("Agent reached the maximum of {0} steps without a final answer")]
    MaxSteps(usize),

    #[error("Provider '{provider}' error{}: {message}", .status.map(|s| format!(" (HTTP {s})")).unwrap_or_default())]
    Provider {
        provider: String,
        message: String,
        status: Option<u16>,
    },

    #[error("Provider '{0}' does not support streaming")]
    StreamingNotSupported(String),
}

impl AgentError {
    pub fn provider(
        provider: impl Into<String>,
        message: impl Into<String>,
        status: Option<u16>,
    ) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
            status,
        }
    }
    pub fn invalid(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidResponse {
            provider: provider.into(),
            message: message.into(),
        }
    }
    pub fn tool_exec(tool: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ToolExecution {
            tool: tool.into(),
            reason: reason.into(),
        }
    }
    pub fn is_client_error(&self) -> bool {
        matches!(self, Self::Provider { status: Some(s), .. } if *s >= 400 && *s < 500)
    }
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network(_)
                | Self::Provider {
                    status: Some(500..=599),
                    ..
                }
        )
    }
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
//
// stream_complete() defaults to returning StreamingNotSupported.
// Providers that support streaming override it and set supports_streaming()
// to return true. All existing providers compile without changes.
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

    /// Streaming completion — yields text chunks as they arrive.
    ///
    /// Only called when no tools are registered. The agent loop uses
    /// complete() when tools are present (tool calls can't be streamed
    /// because parsing requires the full JSON response).
    ///
    /// Default: returns StreamingNotSupported. Override to enable streaming.
    async fn stream_complete(
        &self,
        messages: &[Message],
        model: &str,
    ) -> Result<TokenStream, AgentError> {
        let _ = (messages, model);
        Err(AgentError::StreamingNotSupported(
            self.provider_name().to_string(),
        ))
    }

    /// Returns true if this provider implements stream_complete().
    fn supports_streaming(&self) -> bool {
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports
// ─────────────────────────────────────────────────────────────────────────────

pub use providers::anthropic::AnthropicProvider;
pub use providers::ollama::OllamaProvider;
pub use providers::openai::OpenAiProvider;
pub use providers::openrouter::OpenRouterProvider;

// ─────────────────────────────────────────────────────────────────────────────
// SSE parsing helper (shared across OpenAI-compatible streaming providers)
//
// OpenAI SSE format:
//   data: {"choices":[{"delta":{"content":"hello"}}]}
//   data: [DONE]
// ─────────────────────────────────────────────────────────────────────────────

pub fn parse_sse_chunk(line: &str) -> Option<String> {
    let data = line.strip_prefix("data: ")?;
    if data.trim() == "[DONE]" {
        return None;
    }
    let json: Value = serde_json::from_str(data).ok()?;
    json["choices"][0]["delta"]["content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Tools
// ─────────────────────────────────────────────────────────────────────────────

pub struct AddNumbersTool;
#[async_trait]
impl Tool for AddNumbersTool {
    fn name(&self) -> &'static str {
        "add_numbers"
    }
    fn description(&self) -> &'static str {
        "Adds two integers and returns the result"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false})
    }
    async fn execute(&self, args: Value) -> Result<String, AgentError> {
        let a = args["a"].as_i64().ok_or_else(|| {
            AgentError::tool_exec(
                self.name(),
                "missing or invalid field 'a' (expected integer)",
            )
        })?;
        let b = args["b"].as_i64().ok_or_else(|| {
            AgentError::tool_exec(
                self.name(),
                "missing or invalid field 'b' (expected integer)",
            )
        })?;
        Ok((a + b).to_string())
    }
}

pub struct MultiplyNumbersTool;
#[async_trait]
impl Tool for MultiplyNumbersTool {
    fn name(&self) -> &'static str {
        "multiply_numbers"
    }
    fn description(&self) -> &'static str {
        "Multiplies two integers and returns the result"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false})
    }
    async fn execute(&self, args: Value) -> Result<String, AgentError> {
        let a = args["a"].as_i64().ok_or_else(|| {
            AgentError::tool_exec(
                self.name(),
                "missing or invalid field 'a' (expected integer)",
            )
        })?;
        let b = args["b"].as_i64().ok_or_else(|| {
            AgentError::tool_exec(
                self.name(),
                "missing or invalid field 'b' (expected integer)",
            )
        })?;
        Ok((a * b).to_string())
    }
}

pub struct JokeTool;
#[async_trait]
impl Tool for JokeTool {
    fn name(&self) -> &'static str {
        "get_joke"
    }
    fn description(&self) -> &'static str {
        "Fetches a random family-friendly joke and returns it"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{},"additionalProperties":false})
    }
    async fn execute(&self, _args: Value) -> Result<String, AgentError> {
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

    // ── Blocking run (unchanged) ──────────────────────────────────────────

    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError> {
        self.history.push(Message::user(user_input));
        let mut executed_tool_calls = HashSet::new();

        for step in 0..self.max_steps {
            let tool_refs: Vec<&dyn Tool> = self.tools.iter().map(|t| t.as_ref()).collect();
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
                .map_err(|e| match e {
                    AgentError::Network(inner) => AgentError::Provider {
                        provider: self.provider.provider_name().to_string(),
                        message: format!("network error at step {step}: {inner}"),
                        status: None,
                    },
                    other => other,
                })?;

            let content = completion.content.clone().unwrap_or_default();
            let tool_calls = completion.tool_calls.clone();
            let raw_tool_calls = completion.raw_tool_calls.clone().unwrap_or(Value::Null);
            self.history.push(Message::assistant_with_tools(
                content.clone(),
                raw_tool_calls,
            ));

            if tool_calls.is_empty() {
                if !content.is_empty() {
                    return Ok(content);
                }
                return Err(AgentError::provider(
                    self.provider.provider_name(),
                    "model returned an empty response with no tool calls",
                    None,
                ));
            }

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
                if !content.is_empty() {
                    return Ok(content);
                }
                return Err(AgentError::provider(
                    self.provider.provider_name(),
                    "model repeated already-executed tool calls with no new content",
                    None,
                ));
            }
        }
        Err(AgentError::MaxSteps(self.max_steps))
    }

    // ── Streaming run ─────────────────────────────────────────────────────
    //
    // Returns a TokenStream of text chunks as they arrive from the LLM.
    //
    // When tools are registered or the provider doesn't support streaming,
    // falls back to complete() and wraps the result in a single-item stream
    // — so callers always get a consistent stream interface.
    //
    // Example:
    //   use futures::StreamExt;
    //   let mut stream = agent.stream("Tell me a story").await?;
    //   while let Some(chunk) = stream.next().await {
    //       print!("{}", chunk?);
    //   }

    pub async fn stream(&mut self, user_input: &str) -> Result<TokenStream, AgentError> {
        self.history.push(Message::user(user_input));

        let mut messages = vec![Message {
            role: Role::User,
            content: format!("[SYSTEM]: {}", self.system_prompt),
            tool_call_id: None,
            tool_calls: None,
        }];
        messages.extend(self.history.clone());

        // Fall back to complete() when tools are registered or streaming unsupported
        if !self.tools.is_empty() || !self.provider.supports_streaming() {
            let tool_refs: Vec<&dyn Tool> = self.tools.iter().map(|t| t.as_ref()).collect();
            let completion = self
                .provider
                .complete(&messages, &tool_refs, &self.model)
                .await?;
            let content = completion.content.unwrap_or_default();
            self.history.push(Message::assistant(content.clone()));
            let stream = futures::stream::once(async move { Ok(content) });
            return Ok(Box::pin(stream));
        }

        // True streaming path
        self.provider.stream_complete(&messages, &self.model).await
    }

    // ── stream_collect() — stream and collect into a String ───────────────
    //
    // Convenience wrapper: streams chunks to stdout and returns the full text.
    // Adds the completed response to history automatically.
    //
    // Example:
    //   let answer = agent.stream_collect("Tell me a story").await?;
    //   println!("Full answer: {}", answer);

    pub async fn stream_collect(&mut self, user_input: &str) -> Result<String, AgentError> {
        use futures::StreamExt;

        let mut stream = self.stream(user_input).await?;
        let mut full = String::new();

        while let Some(chunk) = stream.next().await {
            let text = chunk?;
            print!("{}", text);
            full.push_str(&text);
        }
        println!();

        // Add the completed response to history
        self.history.push(Message::assistant(full.clone()));
        Ok(full)
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
