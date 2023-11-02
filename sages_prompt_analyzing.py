from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain


template_prompt = """Review: Great place for pancakes and hot drinks. We received very good service from the staff and we liked the ambiance. I ordered the Salmon with pasta (tagliatelle) and was delighted.
Review Analysis: The review provides a positive rating of the restaurant, with the customer commending the food, service, and ambience. The reviewer specifically mentions ordering the salmon with pasta, indicating that they were satisfied with the dish.
Classification: Positive

Review: {review}
Review analysis: """

review = "Perhaps it was tasty and good service in the past, but since they got so popular, they don’t care at all about the service."


template = PromptTemplate(
    template=template_prompt,
    input_variables=['review']
)

openai = OpenAI(
    model_name='text-davinci-003'
)

llm_chain = LLMChain(prompt=template, llm=openai)

response = llm_chain.run(
    review=review,
    max_tokens=50,
    temperature=0
)

print(response)