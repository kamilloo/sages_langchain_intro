import openai

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Litwo ojczyzno",
    max_tokens=100,
    temperature=1.0,
    n=5,
    stop="\n"
)

for completion in response.choices:
    print(completion.text)
    print('-'*20)