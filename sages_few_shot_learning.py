from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain import LLMChain

llm = OpenAI(
    model_name='text-davinci-003'
)

examples = [
    {
        "review": "Great place for pancakes and hot drinks. We received very good service from the staff and we liked the ambiance. I ordered the Salmon with pasta (tagliatelle) and was delighted.",
        "analysis": "The review provides a positive rating of the restaurant, with the customer commending the food, service, and ambience. The reviewer specifically mentions ordering the salmon with pasta, indicating that they were satisfied with the dish.",
        "stars": "*****",
        "classification": "positive"
    },

    {
        "review": "I ate two dish dinner, The soup was tasty but meat and potatoes not so much",
        "analysis": "The review provides a neutral score of the restaurant, with the customer commending both parts of dinner. The reviewer specifically mentions that first dish was good and second dish not enough good.",
        "stars": "***",
        "classification": "neutral"
    },

    {
        "review": "I eat cold dinner, I hate this place",
        "analysis": "The review provides a negative rating of the restaurant, with the customer commending the food, service, and ambience. The reviewer specifically mentions ordering the salmon with pasta, indicating that they were satisfied with the dish.",
        "stars": "*",
        "classification": "negative"
    },

]

example_template = """Review: {review}
Review Analysis: {analysis}
Stars: {stars}
Classification: {classification}"""

example_prompt = PromptTemplate(
    template=example_template,
    input_variables=['review', 'analysis', 'stars', 'classification']
)

shots_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=['review'],
    suffix="""Review {review}
Review Analysis: """
)

review = "Probably, meat is fresh and good cooked."

llm_chain = LLMChain(prompt=shots_template, llm=llm)

response = llm_chain.run(
    review=review,
    max_tokens=50,
    temperature=0.8
)

print(response)