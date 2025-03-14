from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

API_KEY = "AIzaSyAgRy8gTnQSGioorDXy-ZJSATvbk_34ynU"

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=API_KEY, temperature=0.7)

template = "Generate 10 baby names that starts with {letter}"

prompt_template = PromptTemplate.from_template(template)
prompt = prompt_template.format(letter="B")
response = llm.invoke(prompt).content
print(response)

# Another way to do the same thing using the pipe operator "|"

prompt_template = "You are the best teacher in the world. Explain the {topic} in the easiest and shortest explanation."
prompt = PromptTemplate(input_variables=['topic'], template=prompt_template)
chain = prompt | llm
result = chain.invoke({'topic':'AI'}).content
print(result)