🛠️ Mini‑Agent

A minimal AI agent framework in Rust for building LLM-powered agents with tool execution. Designed for simplicity, flexibility, and easy Rust integration.

Mini‑Agent is for developers who want a lightweight, Rust-native agent library, without depending on large AI frameworks.

🔹 Features

LLM provider abstraction – plug in OpenAI, OpenRouter, or any provider

Tool system – define, register, and invoke tools with structured JSON input/output

Agent loop – simple ReAct-style agent that can call tools and handle outputs

Async ready – fully asynchronous tool execution

Minimal dependencies – small, clean Rust library

📦 Installation

Add this to your Cargo.toml:

[dependencies]
mini-agent = { git = "https://github.com/RajMandaliya/mini-agent" }
⚡ Quick Start
use mini_agent::{
    agent::Agent,
    provider::OpenRouterProvider,
    tool::{Tool, AddNumbersTool}
};

#[tokio::main]
async fn main() {
    // Initialize your LLM provider
    let provider = OpenRouterProvider::new("YOUR_API_KEY");

    // Create an agent with tools
    let mut agent = Agent::new(provider)
        .with_tool(Box::new(AddNumbersTool));

    // Run the agent with a prompt
    let result = agent.run("Add 4 and 7").await.unwrap();

    println!("Agent output: {}", result);
}
🛠️ Creating Your Own Tools
use mini_agent::tool::Tool;
use async_trait::async_trait;
use serde_json::Value;

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
🌐 Contributing

Contributions are welcome!

Add new tools

Improve provider integrations

Add memory/persistence or multi-agent orchestration

Steps:

Fork the repo

Create a new branch

Submit a pull request

📄 License

MIT License – see LICENSE for details.