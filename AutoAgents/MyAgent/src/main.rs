use autoagents::async_trait;
use autoagents::core::agent::AgentOutputT;
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents_derive::{tool, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;





#[derive(Serialize, Deserialize, ToolInput, Debug)]
struct AddArgs {
    #[input(description = "Left operand for addition")]
    left: i64,
    #[input(description = "Right operand for addition")]
    right: i64,
}

#[tool(name = "addition", description = "Add two numbers", input = AddArgs)]
struct Addition;

#[async_trait]
impl ToolRuntime for Addition {
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let a: AddArgs = serde_json::from_value(args)?;
        Ok((a.left + a.right).into())
    }
}

use autoagents_derive::AgentOutput;
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
struct MathOut {
    #[output(description = "The result value")] value: i64,
    #[output(description = "Short explanation")] explanation: String,
}


use autoagents_derive::{agent, AgentHooks};
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};

#[agent(
    name = "math_agent",
    description = "Solve basic math using tools and return JSON",
    tools = [Addition],
    output = MathOut
)]
#[derive(Clone, AgentHooks, Default)]
struct MathAgent;

impl From<ReActAgentOutput> for MathOut {
    fn from(out: ReActAgentOutput) -> Self {
        serde_json::from_str(&out.response).unwrap_or(MathOut { value: 0, explanation: out.response })
    }
}


use autoagents::core::agent::{AgentBuilder, DirectAgent};
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::task::Task;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::backends::deepseek::DeepSeek;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm: Arc<DeepSeek> = LLMBuilder::<DeepSeek>::new()
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .model("deepseek-v4-flash")
        .build()?;

    let agent = ReActAgent::new(MathAgent);
    let handle = AgentBuilder::<_, DirectAgent>::new(agent)
        .llm(llm)
        .memory(Box::new(SlidingWindowMemory::new(10)))
        .build()
        .await?;

    let out = handle.agent.run(Task::new("Add 20 and 5 and explain"))
        .await?;
    println!("{:?}", out);
    Ok(())
}