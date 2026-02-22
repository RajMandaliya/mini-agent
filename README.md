# Mini-Agent (Rust)

A **minimal, extensible AI agent framework** in Rust — composable, async-first, and designed for tool-integrated LLM workflows.

Mini-Agent focuses on predictable structure, simple abstractions, and clean separation of concerns between providers, agents, and tools.

It is built for developers who want a Rust-native agent core without heavy frameworks or hidden complexity.

---

## Motivation

Modern AI agents rely on large language models *and* external tools to complete real-world tasks.  

Most Rust libraries in this space are either experimental, incomplete, or tightly coupled to specific providers.

Mini-Agent aims to provide:

- A clean and understandable agent loop  
- A provider abstraction layer  
- Structured tool execution via JSON schema  
- Async-first design  
- Extensibility without magic  

This project prioritizes **clarity over cleverness** and **architecture over hype**.

---

## Features

- LLM provider abstraction
- Tool registration and execution
- JSON-schema based tool interface
- Async execution model
- ReAct-style agent loop
- Simple and composable API surface

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mini-agent = { git = "https://github.com/RajMandaliya/mini-agent" }
```
Architecture
Core Components
1. Provider
- Wraps the LLM API (OpenRouter by default).
2. Tool
- Defines executable logic with an input schema and structured output.
3. Agent
- Orchestrates conversation flow, tool invocation, and result integration.


Execution Flow
1. user submits a prompt
2. Agent queries the LLM via Provider
3. LLM may request a tool call
4. Agent executes the tool
5. Tool output is fed back into context
6. Loop continues until completion
   
This implements a structured plan → act → observe cycle.



Example (Single File)

Below is a complete example showing:

- Defining a custom tool

- Registering tools

- Running the agent

```rust
  use mini_agent::{
    agent::Agent,
    provider::OpenRouterProvider,
    tool::{Tool, AddNumbersTool}
};
use async_trait::async_trait;
use serde_json::Value;

// --- Custom Tool Definition ---
pub struct MultiplyTool;

#[async_trait]
impl Tool for MultiplyTool {
    fn name(&self) -> &'static str { "MultiplyTool" }

    fn description(&self) -> &'static str {
        "Multiplies two numbers"
    }

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

        serde_json::json!({
            "result": x * y
        })
    }
}

#[tokio::main]
async fn main() {
    let provider = OpenRouterProvider::new("YOUR_API_KEY");

    let mut agent = Agent::new(provider)
        .with_tool(Box::new(AddNumbersTool))
        .with_tool(Box::new(MultiplyTool));

    let result = agent
        .run("Add 4 and 7, then multiply 3 and 5")
        .await
        .unwrap();

    println!("Agent output: {}", result);
}
```

Roadmap

- Memory / persistence layer
- Streaming response support
- Additional provider implementations (OpenAI, Anthropic, etc.)
- Multi-agent orchestration
- Tool registry improvements

Contributing

- Contributions are welcome.
- Improve provider support.
- Add tools
- Expand agent capabilities
- Improve documentation and tests
- Open a PR with a clear description of the change.


License

- MIT License — see the LICENSE file for details.