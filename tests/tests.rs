/// tests/tests.rs — Test suite for mini-agent
///
/// Run with: cargo test
/// Integration tests (requires API key): cargo test --test integration
///
/// Unit tests cover:
///   - Tool trait implementations (AddNumbersTool, MultiplyNumbersTool, JokeTool)
///   - Message construction helpers
///   - Agent builder / configuration
///   - Provider helpers (build_openai_messages, build_openai_tools, parse_openai_completion)
///   - Agent error handling and loop logic via a mock provider

#[cfg(test)]
mod tool_tests {
    use mini_agent::{AddNumbersTool, AgentError, JokeTool, MultiplyNumbersTool, Tool};
    use serde_json::json;

    // ── AddNumbersTool ────────────────────────────────────────────────────

    #[tokio::test]
    async fn add_numbers_basic() {
        let tool = AddNumbersTool;
        let result = tool.execute(json!({ "a": 10, "b": 20 })).await.unwrap();
        assert_eq!(result, "30");
    }

    #[tokio::test]
    async fn add_numbers_negative() {
        let tool = AddNumbersTool;
        let result = tool.execute(json!({ "a": -5, "b": 3 })).await.unwrap();
        assert_eq!(result, "-2");
    }

    #[tokio::test]
    async fn add_numbers_zero() {
        let tool = AddNumbersTool;
        let result = tool.execute(json!({ "a": 0, "b": 0 })).await.unwrap();
        assert_eq!(result, "0");
    }

    #[tokio::test]
    async fn add_numbers_missing_a_returns_error() {
        let tool = AddNumbersTool;
        let result = tool.execute(json!({ "b": 5 })).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolExecution { tool, reason } => {
                assert_eq!(tool, "add_numbers");
                assert!(reason.contains("'a'"));
            }
            _ => panic!("Expected ToolExecution"),
        }
    }

    #[tokio::test]
    async fn add_numbers_missing_b_returns_error() {
        let tool = AddNumbersTool;
        let result = tool.execute(json!({ "a": 5 })).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolExecution { tool, reason } => {
                assert_eq!(tool, "add_numbers");
                assert!(reason.contains("'b'"));
            }
            _ => panic!("Expected ToolExecution"),
        }
    }

    #[tokio::test]
    async fn add_numbers_large_values() {
        let tool = AddNumbersTool;
        let result = tool.execute(json!({ "a": 1_000_000, "b": 2_000_000 })).await.unwrap();
        assert_eq!(result, "3000000");
    }

    // ── MultiplyNumbersTool ───────────────────────────────────────────────

    #[tokio::test]
    async fn multiply_numbers_basic() {
        let tool = MultiplyNumbersTool;
        let result = tool.execute(json!({ "a": 7, "b": 8 })).await.unwrap();
        assert_eq!(result, "56");
    }

    #[tokio::test]
    async fn multiply_numbers_by_zero() {
        let tool = MultiplyNumbersTool;
        let result = tool.execute(json!({ "a": 999, "b": 0 })).await.unwrap();
        assert_eq!(result, "0");
    }

    #[tokio::test]
    async fn multiply_numbers_negative() {
        let tool = MultiplyNumbersTool;
        let result = tool.execute(json!({ "a": -3, "b": 4 })).await.unwrap();
        assert_eq!(result, "-12");
    }

    #[tokio::test]
    async fn multiply_numbers_both_negative() {
        let tool = MultiplyNumbersTool;
        let result = tool.execute(json!({ "a": -3, "b": -4 })).await.unwrap();
        assert_eq!(result, "12");
    }

    #[tokio::test]
    async fn multiply_numbers_missing_a_returns_error() {
        let tool = MultiplyNumbersTool;
        let result = tool.execute(json!({ "b": 5 })).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolExecution { tool, reason } => {
                assert_eq!(tool, "multiply_numbers");
                assert!(reason.contains("'a'"));
            }
            _ => panic!("Expected ToolExecution"),
        }
    }

    #[tokio::test]
    async fn multiply_numbers_missing_b_returns_error() {
        let tool = MultiplyNumbersTool;
        let result = tool.execute(json!({ "a": 5 })).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolExecution { tool, reason } => {
                assert_eq!(tool, "multiply_numbers");
                assert!(reason.contains("'b'"));
            }
            _ => panic!("Expected ToolExecution"),
        }
    }

    // ── Tool metadata ─────────────────────────────────────────────────────

    #[test]
    fn add_tool_metadata() {
        let tool = AddNumbersTool;
        assert_eq!(tool.name(), "add_numbers");
        assert!(!tool.description().is_empty());
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["a"].is_object());
        assert!(schema["properties"]["b"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("a")));
        assert!(schema["required"].as_array().unwrap().contains(&json!("b")));
    }

    #[test]
    fn multiply_tool_metadata() {
        let tool = MultiplyNumbersTool;
        assert_eq!(tool.name(), "multiply_numbers");
        assert!(!tool.description().is_empty());
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn joke_tool_metadata() {
        let tool = JokeTool;
        assert_eq!(tool.name(), "get_joke");
        assert!(!tool.description().is_empty());
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message construction tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod message_tests {
    use mini_agent::{Message, Role};
    use serde_json::json;

    #[test]
    fn user_message_construction() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
        assert!(msg.tool_call_id.is_none());
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn assistant_message_construction() {
        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hi there");
        assert!(msg.tool_call_id.is_none());
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn assistant_with_tools_construction() {
        let calls = json!([{
            "id": "call_1",
            "type": "function",
            "function": { "name": "add_numbers", "arguments": "{\"a\":1,\"b\":2}" }
        }]);
        let msg = Message::assistant_with_tools("", calls.clone());
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.tool_calls, Some(calls));
    }

    #[test]
    fn tool_message_construction() {
        let msg = Message {
            role: Role::Tool,
            content: "42".to_string(),
            tool_call_id: Some("call_1".to_string()),
            tool_calls: None,
        };
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.content, "42");
        assert_eq!(msg.tool_call_id, Some("call_1".to_string()));
    }

    #[test]
    fn role_display() {
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
        assert_eq!(Role::Tool.to_string(), "tool");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent configuration tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod agent_tests {
    use mini_agent::{
        Agent, AddNumbersTool, MultiplyNumbersTool, AgentError, Completion,
        LlmProvider, Message, Tool,
    };
    use async_trait::async_trait;
    use serde_json::json;

    // ── Mock provider that returns a fixed text response ──────────────────

    struct MockProvider {
        response: String,
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        fn provider_name(&self) -> &str { "Mock" }

        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[&dyn Tool],
            _model: &str,
        ) -> Result<Completion, AgentError> {
            Ok(Completion {
                content: Some(self.response.clone()),
                tool_calls: vec![],
                raw_tool_calls: None,
            })
        }
    }

    // ── Mock provider that always returns a provider error ────────────────

    struct ErrorProvider;

    #[async_trait]
    impl LlmProvider for ErrorProvider {
        fn provider_name(&self) -> &str { "ErrorMock" }

        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[&dyn Tool],
            _model: &str,
        ) -> Result<Completion, AgentError> {
            Err(AgentError::provider("ErrorMock", "Simulated provider failure", Some(500)))
        }
    }

    // ── Mock provider that returns empty content with no tool calls ───────

    struct EmptyProvider;

    #[async_trait]
    impl LlmProvider for EmptyProvider {
        fn provider_name(&self) -> &str { "EmptyMock" }

        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[&dyn Tool],
            _model: &str,
        ) -> Result<Completion, AgentError> {
            Ok(Completion {
                content: None,
                tool_calls: vec![],
                raw_tool_calls: None,
            })
        }
    }

    // ── Mock provider that calls a tool once, then returns a final answer ─

    struct ToolCallingProvider {
        call_count: std::sync::Arc<std::sync::Mutex<usize>>,
    }

    #[async_trait]
    impl LlmProvider for ToolCallingProvider {
        fn provider_name(&self) -> &str { "ToolCallingMock" }

        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[&dyn Tool],
            _model: &str,
        ) -> Result<Completion, AgentError> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            if *count == 1 {
                Ok(Completion {
                    content: None,
                    tool_calls: vec![mini_agent::ToolCall {
                        id: "call_abc".to_string(),
                        name: "add_numbers".to_string(),
                        args: json!({ "a": 10, "b": 20 }),
                    }],
                    raw_tool_calls: Some(json!([{
                        "id": "call_abc",
                        "type": "function",
                        "function": { "name": "add_numbers", "arguments": "{\"a\":10,\"b\":20}" }
                    }])),
                })
            } else {
                Ok(Completion {
                    content: Some("The answer is 30".to_string()),
                    tool_calls: vec![],
                    raw_tool_calls: None,
                })
            }
        }
    }

    // ── Mock provider that loops forever (hits max_steps) ─────────────────

    struct InfiniteToolProvider;

    #[async_trait]
    impl LlmProvider for InfiniteToolProvider {
        fn provider_name(&self) -> &str { "InfiniteMock" }

        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[&dyn Tool],
            _model: &str,
        ) -> Result<Completion, AgentError> {
            // Always requests a tool — never gives a final answer
            Ok(Completion {
                content: None,
                tool_calls: vec![mini_agent::ToolCall {
                    id: format!("call_{}", uuid_stub()),
                    name: "add_numbers".to_string(),
                    args: json!({ "a": 1, "b": 1 }),
                }],
                raw_tool_calls: Some(json!([])),
            })
        }
    }

    // Simple unique-ID stub so each tool call gets a fresh ID (avoids the
    // duplicate-call-id short-circuit in the agent loop).
    fn uuid_stub() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        CTR.fetch_add(1, Ordering::Relaxed)
    }

    // ── Agent configuration ───────────────────────────────────────────────

    #[test]
    fn agent_default_config() {
        let provider = MockProvider { response: "hi".into() };
        let agent = Agent::new(Box::new(provider), "test-model");
        assert_eq!(agent.model, "test-model");
        assert_eq!(agent.max_steps, 6);
        assert!(agent.history.is_empty());
        assert!(agent.tools.is_empty());
        assert!(!agent.system_prompt.is_empty());
    }

    #[test]
    fn agent_with_max_steps() {
        let provider = MockProvider { response: "hi".into() };
        let agent = Agent::new(Box::new(provider), "test-model").with_max_steps(20);
        assert_eq!(agent.max_steps, 20);
    }

    #[test]
    fn agent_with_system_prompt() {
        let provider = MockProvider { response: "hi".into() };
        let agent = Agent::new(Box::new(provider), "test-model")
            .with_system_prompt("Custom prompt here");
        assert_eq!(agent.system_prompt, "Custom prompt here");
    }

    #[test]
    fn agent_add_tool_increases_count() {
        let provider = MockProvider { response: "hi".into() };
        let mut agent = Agent::new(Box::new(provider), "test-model");
        assert_eq!(agent.tools.len(), 0);
        agent.add_tool(AddNumbersTool);
        assert_eq!(agent.tools.len(), 1);
        agent.add_tool(MultiplyNumbersTool);
        assert_eq!(agent.tools.len(), 2);
    }

    // ── Agent run ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn agent_run_returns_text_response() {
        let provider = MockProvider { response: "Hello from mock!".into() };
        let mut agent = Agent::new(Box::new(provider), "test-model");
        let result = agent.run("Say hello").await.unwrap();
        assert_eq!(result, "Hello from mock!");
    }

    #[tokio::test]
    async fn agent_run_appends_to_history() {
        let provider = MockProvider { response: "42".into() };
        let mut agent = Agent::new(Box::new(provider), "test-model");
        agent.run("What is 6 x 7?").await.unwrap();
        assert!(agent.history.len() >= 2); // at minimum: user + assistant
        assert_eq!(agent.history[0].content, "What is 6 x 7?");
    }

    #[tokio::test]
    async fn agent_run_propagates_provider_error() {
        let mut agent = Agent::new(Box::new(ErrorProvider), "test-model");
        let result = agent.run("anything").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::Provider { provider, message, status } => {
                assert_eq!(provider, "ErrorMock");
                assert!(message.contains("Simulated provider failure"));
                assert_eq!(status, Some(500));
            }
            _ => panic!("Expected Provider error"),
        }
    }

    #[tokio::test]
    async fn agent_run_empty_response_returns_error() {
        let mut agent = Agent::new(Box::new(EmptyProvider), "test-model");
        let result = agent.run("anything").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::Provider { message, .. } => {
                assert!(message.contains("empty response"));
            }
            _ => panic!("Expected Provider error for empty response"),
        }
    }

    #[tokio::test]
    async fn agent_executes_tool_and_returns_answer() {
        let call_count = std::sync::Arc::new(std::sync::Mutex::new(0));
        let provider = ToolCallingProvider { call_count: call_count.clone() };
        let mut agent = Agent::new(Box::new(provider), "test-model");
        agent.add_tool(AddNumbersTool);

        let result = agent.run("Add 10 and 20").await.unwrap();
        assert_eq!(result, "The answer is 30");

        // Provider should be called twice: once for tool call, once for final answer
        assert_eq!(*call_count.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn agent_history_includes_tool_result() {
        let call_count = std::sync::Arc::new(std::sync::Mutex::new(0));
        let provider = ToolCallingProvider { call_count };
        let mut agent = Agent::new(Box::new(provider), "test-model");
        agent.add_tool(AddNumbersTool);

        agent.run("Add 10 and 20").await.unwrap();

        // History should contain: user, assistant (tool call), tool (result), assistant (final)
        let roles: Vec<String> = agent.history.iter().map(|m| m.role.to_string()).collect();
        assert!(roles.contains(&"user".to_string()));
        assert!(roles.contains(&"tool".to_string()));
        assert!(roles.contains(&"assistant".to_string()));
    }

    #[tokio::test]
    async fn agent_tool_not_found_returns_error() {
        // Provider requests a tool that isn't registered on the agent
        let provider = ToolCallingProvider {
            call_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
        };
        let mut agent = Agent::new(Box::new(provider), "test-model");
        // AddNumbersTool deliberately not registered

        let result = agent.run("Add 10 and 20").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolNotFound(name) => assert_eq!(name, "add_numbers"),
            _ => panic!("Expected ToolNotFound"),
        }
    }

    #[tokio::test]
    async fn agent_hits_max_steps_returns_error() {
        let mut agent = Agent::new(Box::new(InfiniteToolProvider), "test-model")
            .with_max_steps(3);
        agent.add_tool(AddNumbersTool);

        let result = agent.run("loop forever").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::MaxSteps(n) => assert_eq!(n, 3),
            _ => panic!("Expected MaxSteps error"),
        }
    }

    #[tokio::test]
    async fn agent_error_is_retryable_for_500() {
        let err = AgentError::provider("SomeProvider", "internal server error", Some(500));
        assert!(err.is_retryable());
        assert!(!err.is_client_error());
    }

    #[tokio::test]
    async fn agent_error_is_client_error_for_401() {
        let err = AgentError::provider("SomeProvider", "invalid api key", Some(401));
        assert!(err.is_client_error());
        assert!(!err.is_retryable());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider helper tests (parse_openai_completion, build_openai_messages, etc.)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod provider_helper_tests {
    use mini_agent::providers::{build_openai_messages, build_openai_tools, parse_openai_completion};
    use mini_agent::{AgentError, Message, Role, Tool};
    use async_trait::async_trait;
    use serde_json::{json, Value};

    struct DummyTool;

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &'static str { "dummy_tool" }
        fn description(&self) -> &'static str { "A dummy tool for testing" }
        fn parameters_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "x": { "type": "integer" } },
                "required": ["x"]
            })
        }
        async fn execute(&self, _args: Value) -> Result<String, AgentError> {
            Ok("dummy_result".to_string())
        }
    }

    // ── build_openai_messages ─────────────────────────────────────────────

    #[test]
    fn build_messages_user_only() {
        let messages = vec![Message::user("Hello")];
        let result = build_openai_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "Hello");
    }

    #[test]
    fn build_messages_includes_tool_call_id() {
        let messages = vec![Message {
            role: Role::Tool,
            content: "result_value".to_string(),
            tool_call_id: Some("call_xyz".to_string()),
            tool_calls: None,
        }];
        let result = build_openai_messages(&messages);
        assert_eq!(result[0]["tool_call_id"], "call_xyz");
    }

    #[test]
    fn build_messages_omits_null_tool_calls() {
        let messages = vec![Message::user("hi")];
        let result = build_openai_messages(&messages);
        assert!(result[0].get("tool_calls").is_none());
    }

    #[test]
    fn build_messages_preserves_order() {
        let messages = vec![
            Message::user("first"),
            Message::assistant("second"),
            Message::user("third"),
        ];
        let result = build_openai_messages(&messages);
        assert_eq!(result[0]["content"], "first");
        assert_eq!(result[1]["content"], "second");
        assert_eq!(result[2]["content"], "third");
    }

    // ── build_openai_tools ────────────────────────────────────────────────

    #[test]
    fn build_tools_correct_shape() {
        let tool = DummyTool;
        let tools: Vec<&dyn Tool> = vec![&tool];
        let result = build_openai_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "function");
        assert_eq!(result[0]["function"]["name"], "dummy_tool");
        assert_eq!(result[0]["function"]["description"], "A dummy tool for testing");
        assert!(result[0]["function"]["parameters"].is_object());
    }

    #[test]
    fn build_tools_empty() {
        let tools: Vec<&dyn Tool> = vec![];
        let result = build_openai_tools(&tools);
        assert!(result.is_empty());
    }

    // ── parse_openai_completion ───────────────────────────────────────────

    #[test]
    fn parse_completion_text_only() {
        let json = json!({
            "choices": [{
                "message": { "role": "assistant", "content": "Hello world" }
            }]
        });
        let completion = parse_openai_completion(&json).unwrap();
        assert_eq!(completion.content, Some("Hello world".to_string()));
        assert!(completion.tool_calls.is_empty());
        assert!(completion.raw_tool_calls.is_none());
    }

    #[test]
    fn parse_completion_with_tool_calls() {
        let json = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "add_numbers",
                            "arguments": "{\"a\":1,\"b\":2}"
                        }
                    }]
                }
            }]
        });
        let completion = parse_openai_completion(&json).unwrap();
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.tool_calls[0].name, "add_numbers");
        assert_eq!(completion.tool_calls[0].id, "call_1");
        assert_eq!(completion.tool_calls[0].args["a"], 1);
        assert_eq!(completion.tool_calls[0].args["b"], 2);
    }

    #[test]
    fn parse_completion_missing_choices_returns_error() {
        let json = json!({ "error": "something went wrong" });
        let result = parse_openai_completion(&json);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::InvalidResponse { message, .. } => {
                assert!(message.contains("choices"));
            }
            _ => panic!("Expected InvalidResponse"),
        }
    }

    #[test]
    fn parse_completion_empty_choices_returns_error() {
        let json = json!({ "choices": [] });
        let result = parse_openai_completion(&json);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::InvalidResponse { .. } => {}
            _ => panic!("Expected InvalidResponse"),
        }
    }

    #[test]
    fn parse_completion_bad_args_json_returns_error() {
        let json = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "some_tool",
                            "arguments": "NOT VALID JSON {{{"
                        }
                    }]
                }
            }]
        });
        let result = parse_openai_completion(&json);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::InvalidResponse { message, .. } => {
                assert!(message.contains("some_tool"));
            }
            _ => panic!("Expected InvalidResponse with tool name context"),
        }
    }

    #[test]
    fn parse_completion_null_content() {
        let json = json!({
            "choices": [{
                "message": { "role": "assistant", "content": null }
            }]
        });
        let completion = parse_openai_completion(&json).unwrap();
        assert!(completion.content.is_none());
    }

    #[test]
    fn parse_completion_missing_function_field_returns_error() {
        let json = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{ "id": "call_1", "type": "function" }]
                    // "function" field is missing entirely
                }
            }]
        });
        let result = parse_openai_completion(&json);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::InvalidResponse { message, .. } => {
                assert!(message.contains("function"));
            }
            _ => panic!("Expected InvalidResponse"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod error_tests {
    use mini_agent::AgentError;

    #[test]
    fn tool_not_found_display() {
        let err = AgentError::ToolNotFound("my_tool".to_string());
        let msg = err.to_string();
        assert!(msg.contains("my_tool"));
        assert!(msg.contains("add_tool"));  // hint text
    }

    #[test]
    fn tool_execution_display() {
        let err = AgentError::tool_exec("my_tool", "bad input");
        let msg = err.to_string();
        assert!(msg.contains("my_tool"));
        assert!(msg.contains("bad input"));
    }

    #[test]
    fn max_steps_display_includes_limit() {
        let err = AgentError::MaxSteps(10);
        let msg = err.to_string();
        assert!(msg.contains("10"));
    }

    #[test]
    fn invalid_response_display_includes_provider() {
        let err = AgentError::invalid("OpenAI", "missing field");
        let msg = err.to_string();
        assert!(msg.contains("OpenAI"));
        assert!(msg.contains("missing field"));
    }

    #[test]
    fn provider_error_display_includes_status() {
        let err = AgentError::provider("Anthropic", "rate limited", Some(429));
        let msg = err.to_string();
        assert!(msg.contains("Anthropic"));
        assert!(msg.contains("rate limited"));
        assert!(msg.contains("429"));
    }

    #[test]
    fn provider_error_display_without_status() {
        let err = AgentError::provider("Ollama", "connection refused", None);
        let msg = err.to_string();
        assert!(msg.contains("Ollama"));
        assert!(msg.contains("connection refused"));
        // No HTTP status in message
        assert!(!msg.contains("HTTP"));
    }

    #[test]
    fn json_error_from_serde() {
        let bad_json = "{ not valid }";
        let serde_err = serde_json::from_str::<serde_json::Value>(bad_json).unwrap_err();
        let agent_err: AgentError = serde_err.into();
        assert!(matches!(agent_err, AgentError::Json(_)));
    }

    #[test]
    fn is_retryable_for_5xx() {
        assert!(AgentError::provider("X", "err", Some(500)).is_retryable());
        assert!(AgentError::provider("X", "err", Some(503)).is_retryable());
    }

    #[test]
    fn is_not_retryable_for_4xx() {
        assert!(!AgentError::provider("X", "err", Some(400)).is_retryable());
        assert!(!AgentError::provider("X", "err", Some(401)).is_retryable());
        assert!(!AgentError::provider("X", "err", Some(404)).is_retryable());
    }

    #[test]
    fn is_client_error_for_4xx() {
        assert!(AgentError::provider("X", "err", Some(400)).is_client_error());
        assert!(AgentError::provider("X", "err", Some(401)).is_client_error());
        assert!(AgentError::provider("X", "err", Some(422)).is_client_error());
    }

    #[test]
    fn is_not_client_error_for_5xx() {
        assert!(!AgentError::provider("X", "err", Some(500)).is_client_error());
    }
}