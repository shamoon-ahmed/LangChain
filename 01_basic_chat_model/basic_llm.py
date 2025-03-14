from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                             api_key=os.getenv('GEMINI_API_KEY'),
                             temperature=0.7)
response = llm.invoke("What is the capital of France?")
print(response.content)

# Using SystemMessage and HumanMessages

messages = [SystemMessage(content='You are the best teacher in the world. Answer the questions in the most understandable way.'),
            HumanMessage('What do you think the future of AI would look like?')]

response = llm.invoke(messages)
print(response.content)