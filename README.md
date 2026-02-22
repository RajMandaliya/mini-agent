# Mini‑Agent (Rust)

A **minimal, extensible AI agent framework** in Rust — composable, async‑first, and built for tool‑integrated LLM workflows.

Mini‑Agent focuses on predictable structure, simple abstractions, and easy integration with different LLM providers and tool executors.

---

## Motivation

Modern AI agents rely on LLMs *and* tools for real‑world tasks. Many Rust libraries today are minimal or incomplete, with limited flexibility.

Mini‑Agent aims to provide a **clean, intuitive agent core** that supports:

- LLM interaction abstraction  
- Tool invocation via JSON schema  
- Async tool execution  
- Structured agent loop for planning & acting

This project prioritizes **clarity over magic** and **extensibility over complexity**.

---

## Status

**Early‑stage but usable.**  
The core agent loop, provider abstraction, and simple tools are implemented, but APIs and feature sets may evolve with feedback and real‑world usage.

Expect improvements around:

- Memory and context persistence  
- Multiple provider support  
- Streaming outputs  
- Expanded tool registry

---

## High‑Level Architecture

### Concepts

- **Provider** — wraps the LLM API (OpenRouter by default)  
- **Tool** — encapsulates an executable action with input schema  
- **Agent** — orchestrates prompt parsing, tool calls, and results

### Workflow

1. User submits a *prompt* to the agent  
2. The agent queries the LLM via the provider  
3. If the LLM *requests a tool call*, the agent executes it  
4. Tool results are fed back into the conversation  
5. The loop continues until completion

---

## Example Usage (Single File)

Below is an example showing **defining a custom tool**, registering it, and running the agent:

```rust
use mini_agent::{
    agent::Agent,
    provider::OpenRouterProvider,
    tool::{Tool, AddNumbersTool}
};
use async_trait::async_trait;
use serde_json::Value;
```
// --- Define a custom tool ---

```
pub struct MultiplyTool;
#[async_trait]
impl Tool for MultiplyTool {
    fn name(&self) -> &'static str { "MultiplyTool" }
    fn description(&self) -> &'static str { "Multiplies two numbers" }
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": "number" },
                "y": { "type": "number" }
            },
            "required": ["x", "y"]
        })
    }

    async fn execute(&self, input: Value) -> Value {
        let x = input["x"].as_i64().unwrap_or(1);
        let y = input["y"].as_i64().unwrap_or(1);
        serde_json::json!({ "result": x * y })
    }
}

```
```
#[tokio::main]
async fn main() {
    // Initialize an LLM provider
    let provider = OpenRouterProvider::new("YOUR_API_KEY");

    // Register built‑in and custom tools
    let mut agent = Agent::new(provider)
        .with_tool(Box::new(AddNumbersTool))
        .with_tool(Box::new(MultiplyTool));

    // Example prompt using tools
    let result = agent
        .run("Add 4 and 7, then multiply 3 and 5")
        .await
        .unwrap();

    println!("Agent output: {}", result);
}
