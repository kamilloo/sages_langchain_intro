import os
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain import LLMChain


CACHE_CATALOG = os.getcwd() + "/.model_cache"


llm = HuggingFacePipeline.from_model_id(
    model_id='gpt2',
    task='text-generation',
    model_kwargs={'temperature': 0.1, "cache_dir":CACHE_CATALOG},
    pipeline_kwargs={'do_sample':True, 'max_new_tokens': 20}
)


template = """Review: {review}

    Classify the review ({options}):"""

prompt_template = PromptTemplate(
    input_variables=["review", "options"],
    template=template
)

review = "Great place for visit, I will be visit many times"

llm_chain = LLMChain(prompt=prompt_template, llm=llm)
response = llm_chain.run(
    review=review,
    options="positive/negative",
    # stop="\n"
)

print(response)