from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper


llm = OpenAI(
    model_name='text-davinci-003'
)

wikipedia = WikipediaAPIWrapper()

tools = [
        Tool.from_function(
        func=wikipedia.run,
        name="Wikipedia",
        description="Useful when you need to find definition of sth"
        )
    ]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

print(agent('What is the smallest island?'))