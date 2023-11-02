from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI


llm = OpenAI(
    model_name='text-davinci-003'
)

tools = load_tools(['llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

print(agent('Third power of 10'))