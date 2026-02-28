# Mini-Agent (Rust)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)
![Rust](https://img.shields.io/badge/rust-async--first-orange.svg)

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

- Multi-provider support (OpenRouter, OpenAI, Anthropic, Ollama)
- Tool registration and execution
- JSON schema based tool interface
- Async execution model
- ReAct-style agent loop
- Configurable system prompt
- Simple and composable API surface

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mini-agent = { git = "https://github.com/RajMandaliya/mini-agent" }
```

---

## Architecture

### Core Components

**Provider**
Wraps the LLM API. Implements the `LlmProvider` trait to send messages and return completions. Built-in providers: `OpenRouterProvider`, `OpenAiProvider`, `AnthropicProvider`, `OllamaProvider`.

**Tool**
Defines executable logic with a JSON schema for inputs and a structured string output. Implement the `Tool` trait to create custom tools.

**Agent**
Orchestrates conversation flow, tool invocation, and result integration. Maintains message history and drives the ReAct loop.

### Execution Flow

```
User prompt
    │
    ▼
Agent sends messages + tools → LLM Provider
    │
    ▼
LLM responds with tool call?
    ├── Yes → Agent executes tool → result added to context → loop
    └── No  → Return final answer
```

This implements a structured **plan → act → observe** cycle.

---

## Example

Below is a complete example showing how to define a custom tool, register tools, and run the agent:

```rust
use mini_agent::{Agent, OpenRouterProvider, AgentError};
use mini_agent::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

// --- Custom Tool Definition ---
pub struct MultiplyTool;

#[async_trait]
impl Tool for MultiplyTool {
    fn name(&self) -> &'static str {
        "multiply_numbers"
    }

    fn description(&self) -> &'static str {
        "Multiplies two integers and returns the result"
    }

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
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        Ok((a * b).to_string())
    }
}

#[tokio::main]
async fn main() {
    let provider = OpenRouterProvider::new("YOUR_API_KEY");

    let mut agent = Agent::new(
        Box::new(provider),
        "mistralai/mistral-7b-instruct:free",
    );

    agent.add_tool(MultiplyTool);

    let result = agent
        .run("Multiply 6 and 7")
        .await
        .unwrap();

    println!("Agent output: {}", result);
}
```

### Custom System Prompt

You can override the default system prompt using the builder:

```rust
let mut agent = Agent::new(Box::new(provider), model)
    .with_system_prompt("You are a math assistant. Only use tools when necessary.");
```

### Max Steps

Control how many reasoning steps the agent can take before giving up:

```rust
let mut agent = Agent::new(Box::new(provider), model)
    .with_max_steps(10);
```

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `AddNumbersTool` | Adds two integers |
| `MultiplyNumbersTool` | Multiplies two integers |
| `JokeTool` | Fetches a random family-friendly joke |

---

## Supported Providers

| Provider | Struct | Free Tier |
|----------|--------|-----------|
| OpenRouter | `OpenRouterProvider` | ✅ Yes |
| OpenAI | `OpenAiProvider` | ❌ Paid |
| Anthropic | `AnthropicProvider` | ❌ Paid |
| Ollama | `OllamaProvider` | ✅ Local |

---

## Example Output

![Mini-Agent Terminal Output](./assets/terminal-output.png)

---

## Testing

Run the test suite with:

```bash
cargo test
```

Unit tests cover core tool logic (add, multiply) and agent error handling. Integration tests require a valid API key set as an environment variable:

```bash
OPENROUTER_API_KEY=your_key cargo test --test integration
```

---

## CI

This project uses GitHub Actions for continuous integration. On every push and pull request, the pipeline runs:

```bash
cargo build
cargo test
cargo clippy
```

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full workflow definition.

---

## Roadmap

- [ ] Memory / persistence layer
- [ ] Streaming response support
- [ ] Multi-agent orchestration
- [ ] Tool registry improvements
- [ ] Expanded test coverage

---

## Contributing

Contributions are welcome. You can:

- Add or improve provider support
- Create new tools
- Expand agent capabilities
- Improve documentation and tests

Open a PR with a clear description of your change.

---

## License

MIT License — see the [LICENSE](./LICENSE) file for details.