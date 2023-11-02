from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain


template_prompt = """{review}
Oceń opinię ({options}, {score}):
I napisz opinię wersję w języku angielskim:"""


template = PromptTemplate(
    template=template_prompt,
    input_variables=['review', 'options', 'score']
)

openai = OpenAI(
    model_name='text-davinci-003'
)

review = "Opinia użytkownika: Jeśli ktoś lubi zatłoczone, hałaśliwe miejsca to w lipcu może się tam wybrać. Zupełnie mi się nie podoba."

# prompt=template.format(review=review, options='ok/middle/negative', score='1-100')
# response = openai(prompt=prompt, max_tokens=50, temperature=0.8)


llm_chain = LLMChain(prompt=template, llm=openai)

response = llm_chain.run(
    review=review,
    options='pozytywna/neutralna/negatywna',
    score='1-100',
    max_tokens=50,
    temperature=0.0,
    # stop="\n"
)

print(response)