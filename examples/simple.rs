/// simple.rs — demonstrates swapping providers with zero agent-code changes.
use mini_agent::{
    Agent, AddNumbersTool, MultiplyNumbersTool, JokeTool,
    // Pick ONE of the four providers below:
    OpenRouterProvider,
    // OpenAiProvider,
    // AnthropicProvider,
    // OllamaProvider,
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // ── Provider selection ─────────────────────────────────────────────────
    //
    // Uncomment the provider you want to use and set the corresponding env var.
    //
    // Option 1: OpenRouter (default — any model via openrouter.ai)
    let api_key = env::var("OPENROUTER_API_KEY")?;
    let model   = "meta-llama/llama-3.1-8b-instruct";
    let provider = OpenRouterProvider::new(api_key, model);

    // Option 2: OpenAI
    // let api_key  = env::var("OPENAI_API_KEY")?;
    // let model    = "gpt-4o-mini";
    // let provider = OpenAiProvider::new(api_key, model);

    // Option 3: Anthropic (Claude)
    // let api_key  = env::var("ANTHROPIC_API_KEY")?;
    // let model    = "claude-sonnet-4-20250514";
    // let provider = AnthropicProvider::new(api_key, model);

    // Option 4: Ollama (local — no API key needed, just `ollama serve`)
    // let model    = "llama3";
    // let provider = OllamaProvider::new(model);

    // ── Agent setup ────────────────────────────────────────────────────────
    let mut agent = Agent::new(Box::new(provider), model);
    agent.add_tool(AddNumbersTool);
    agent.add_tool(MultiplyNumbersTool);
    agent.add_tool(JokeTool);

    // ── Run ────────────────────────────────────────────────────────────────
    let questions = [
        "What is 56 + 89? Answer with just the number.",
        "Multiply 7 and 8 and give only the result.",
        "Tell me one joke.",
    ];

    for question in &questions {
        println!("Question: {}", question);
        match agent.run(question).await {
            Ok(answer) => println!("Answer: {}\n", answer),
            Err(e)     => println!("Error: {}\n", e),
        }
    }

    println!("--- Conversation History ---");
    for msg in &agent.history {
        if !msg.content.trim().is_empty() {
            println!("{} → {}", msg.role, msg.content);
        }
    }

    Ok(())
}