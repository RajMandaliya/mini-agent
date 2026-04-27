/// streaming.rs — demonstrates Agent::stream() and Agent::stream_collect()
///
/// Run with:
///   OPENROUTER_API_KEY=your_key cargo run --example streaming
///   OPENAI_API_KEY=your_key cargo run --example streaming -- --openai
use futures::StreamExt;
use mini_agent::{Agent, OpenRouterProvider};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let use_openai = args.contains(&"--openai".to_string());

    // ── Provider ───────────────────────────────────────────────────────────
    let (provider, model): (Box<dyn mini_agent::LlmProvider>, &str) = if use_openai {
        let key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        (
            Box::new(mini_agent::OpenAiProvider::new(key, "gpt-4o-mini")),
            "gpt-4o-mini",
        )
    } else {
        let key = env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
        (
            Box::new(OpenRouterProvider::new(
                key,
                "meta-llama/llama-3.1-8b-instruct",
            )),
            "meta-llama/llama-3.1-8b-instruct",
        )
    };

    let mut agent =
        Agent::new(provider, model).with_system_prompt("You are a helpful assistant. Be concise.");

    // ── Example 1: stream() — manual chunk handling ────────────────────────
    println!("=== stream() — manual chunk handling ===");
    println!("Prompt: Tell me a two-sentence story about a robot.\n");
    print!("Response: ");

    let mut stream = agent
        .stream("Tell me a two-sentence story about a robot.")
        .await?;
    let mut full_response = String::new();

    while let Some(chunk) = stream.next().await {
        let text = chunk?;
        print!("{}", text);
        full_response.push_str(&text);
    }
    println!("\n");

    // Add to history manually when using stream() directly
    agent
        .history
        .push(mini_agent::Message::assistant(full_response));

    // ── Example 2: stream_collect() — automatic collection ─────────────────
    println!("=== stream_collect() — auto collect + history ===");
    println!("Prompt: What is Rust's ownership model in one sentence?\n");
    print!("Response: ");

    let answer = agent
        .stream_collect("What is Rust's ownership model in one sentence?")
        .await?;

    println!("\nCollected: {}", answer);
    println!("\nHistory entries: {}", agent.history.len());

    Ok(())
}
