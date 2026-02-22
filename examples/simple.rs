use mini_agent::{Agent, AddNumbersTool, OpenRouterProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //dotenvy::dotenv().ok();

    let api_key = std::env::var("OPENROUTER_API_KEY")?;

    let model = "meta-llama/llama-3.1-8b-instruct";
    let provider = OpenRouterProvider::new(api_key, model);

    let mut agent = Agent::new(Box::new(provider), model);
    agent.add_tool(AddNumbersTool);

    let question = "What is 56 + 89? Answer with just the number.";
    println!("Question: {}", question);

    let answer = agent.run(question).await?;
    println!("\nAnswer: {}", answer);

    println!("\nHistory:");
    for msg in &agent.history {
        println!("{} → {}", msg.role, msg.content);
    }

    Ok(())
}