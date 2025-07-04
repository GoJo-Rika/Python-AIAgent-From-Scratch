import json
import re

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from tools import save_tool, search_tool, wiki_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant that will help generate a research paper.\n"
            "Answer the user query and use necessary tools.\n"
            "Wrap the output in this format and provide no other text\n{format_instructions}",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

response_text = raw_response.get("output", "")

try:
    if isinstance(response_text, str):
        cleaned_text = re.sub(
            r"^```(?:json)?\n|```$", "", response_text.strip(), flags=re.MULTILINE
        )
        structured_response = parser.parse(cleaned_text)

        formatted_output = (
            f"Topic: {structured_response.topic}\n\n"
            f"Summary: {structured_response.summary}\n\n"
            f"Sources: {', '.join(structured_response.sources) or 'None'}\n"
            f"Tools Used: {', '.join(structured_response.tools_used) or 'None'}\n"
        )

        result = save_tool(formatted_output)
        print(result)
    else:
        print("Error: 'output' is not a string")

except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)
