from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

wikipedia = WikipediaAPIWrapper()

tools = [
    Tool.from_function(
        func=wikipedia.run,
        name="Wikipedia",
        description="Useful when you need to find a definition of something"
    )
]

llm = OpenAI(
    model_name='text-davinci-003'
)

conversation_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)


agent = initialize_agent(
    tools,
    llm,
    verbose=True,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=conversation_memory,
    max_iterations=3,
    early_stopping_method='generate'
)

print(agent('Find an article about Nile. What is the length of the river?'))

print(agent('The river flows into a sea. What is the area of the sea?'))