from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv() 

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool('get_weather', description = "Return weather information for a given city", return_direct = False)
def get_weather(city: str) -> str:
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


@tool("locate_user", description = "Return the location of the user")
def locate_user(runtime: ToolRuntime) -> str:
    match runtime.context.user_id:
        case "123":
            return "Vienna"
        case "456":
            return "London"
        case _:
            return "Unknown"


model = init_chat_model(model = "gpt-4.1-mini", temperature = 0.7)

checkpointer = InMemorySaver()


agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt = "You are a helpful assistant that can help with weather and location information.",
    checkpointer = checkpointer,
    context_schema=Context,
    response_format=ResponseFormat
)

config = {"configurable": {"thread_id": "123"}}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in Vienna?"}
        ],
        "context": Context(user_id="123")
    },
    config=config
)

print(response['structured_response'])
print(response['structured_response'].temperature_celsius)
print(response['structured_response'].temperature_fahrenheit)
print(response['structured_response'].summary )
