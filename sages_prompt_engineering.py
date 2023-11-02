from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain


template_prompt = """Review: They have very kind and polite workers, i don’t really know about the prices because all i bought was some bottles of beer.
{{ \
"classification": "positive", \
"score": "100" \
}}

Review: Perhaps it was tasty and good service in the past, but since they got so popular, they don’t care at all about the service.
{{ \
"classification": "negative", \
"score": "1" \
}}

Review: {review}
"""

review = "You can eat pancakes and hot drinks. The price are moderate. I ordered the Salmon with pasta (tagliatelle) and was delighted."


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
    temperature=0,
    stop="\n"
)

print(response)

import json
parsed_json = json.loads(response)
print(parsed_json['classification'])
print(parsed_json['score'])