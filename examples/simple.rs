use mini_agent::{Agent, AddNumbersTool, MultiplyNumbersTool, JokeTool, OpenRouterProvider};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("OPENROUTER_API_KEY")?;
    let model = "meta-llama/llama-3.1-8b-instruct";

    let provider = OpenRouterProvider::new(api_key, model);

    let mut agent = Agent::new(Box::new(provider), model);
    agent.add_tool(AddNumbersTool);
    agent.add_tool(MultiplyNumbersTool);
    agent.add_tool(JokeTool);

    // Test addition
    let question1 = "What is 56 + 89? Answer with just the number.";
    println!("Question: {}", question1);
    let answer1 = agent.run(question1).await?;
    println!("Answer: {}\n", answer1);

    // Test multiplication
    let question2 = "Multiply 7 and 8 and give only the result.";
    println!("Question: {}", question2);
    let answer2 = agent.run(question2).await?;
    println!("Answer: {}\n", answer2);

    // Test joke
    let question3 = "Tell me one joke.";
    println!("Question: {}", question3);
    let answer3 = agent.run(question3).await?;
    println!("Joke: {}\n", answer3);

    println!("--- Conversation History ---");
    for msg in &agent.history {
        println!("{} → {}", msg.role, msg.content);
    }

    Ok(())
}