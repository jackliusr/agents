from email.mime import message

from daytona import Daytona
from langchain_daytona import DaytonaSandbox
from deepagents.backends import LocalShellBackend
import csv
import io
import os
from langchain.tools import tool
from slack_sdk import WebClient

from langchain_core.utils.uuid import uuid7

from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

from dotenv import load_dotenv
load_dotenv()

sandbox = Daytona().create();
# backend = DaytonaSandbox(sandbox=sandbox);
backend = LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"})
result = backend.execute("echo ready to analyze data")
print(result)

# Create sample sales data
data = [
    ["Date", "Product", "Units Sold", "Revenue"],
    ["2025-08-01", "Widget A", 10, 250],
    ["2025-08-02", "Widget B", 5, 125],
    ["2025-08-03", "Widget A", 7, 175],
    ["2025-08-04", "Widget C", 3, 90],
    ["2025-08-05", "Widget B", 8, 200],
]

# Convert data to CSV bytes
csv_buffer = io.StringIO()
csv_writer = csv.writer(csv_buffer)
csv_writer.writerows(data)
csv_bytes = csv_buffer.getvalue().encode('utf-8')
csv_buffer.close()

# Upload to backend

backend.upload_files([( "/home/jackl/data/sales_data.csv", csv_bytes )])
print("uploaded sales data to backend")

slack_token = os.environ["SLACK_USER_TOKEN"]
slack_client = WebClient(token=slack_token)
print("connected to Slack successfully")

@tool(parse_docstring=True)
def slack_send_message(text: str, channel: str, file_path: str | None = None) -> str:
    """Send message, optionally including attachments such as images.

    Args:
        text: (str) text content of the message
        channel: (str) the Slack channel to send the message to
        file_path: (str) file path of attachment in the filesystem.
    """
    if not file_path:
        slack_client.chat_postMessage(channel=channel, text=text)
    else:
        fp = backend.download_files([file_path])
        slack_client.files_upload_v2(
            channel="C0AUDTADJ7P", # my own channel
            content=fp[0].content,
            initial_comment=text,)

    return "Message sent to Slack successfully."

checkpointer = InMemorySaver()
agent = create_deep_agent(
    model = "ollama:qwen3.6:35b-a3b-q4_K_M",
    tools=[slack_send_message],
    checkpointer=checkpointer,
    backend=backend,
)

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

input_message = {
    "role": "user",
    "content": (
        "Analyze ./data/sales_data.csv in the current dir and generate a beautiful plot. "
        "When finished, send your analysis and the plot to Slack using the tool."
        )
}

for step in agent.stream(
    {"messages": [input_message]},
     config,
     stream_mode="update",
     ):
    for _, update in step.items():
        if update and (messages := update.get("messages")) and isinstance(messages, list):
            for message in messages:
                message.pretty_print()