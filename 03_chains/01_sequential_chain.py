from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)

template1 = "Define life in 10 words"
template2 = "Try your best to use letter {letter} as the first letter in every word(if possible) in : {promptt}"

prompt_1 = PromptTemplate.from_template(template1)
chain1 = LLMChain(llm=llm, prompt=prompt_1, output_key="promptt")

prompt_2 = PromptTemplate.from_template(template2)
chain2 = LLMChain(llm=llm, prompt=prompt_2, output_key="answer")

chains = SequentialChain(chains=[chain1, chain2],
                         input_variables=['letter'],
                         output_variables=["promptt", "answer"], verbose=True)

inputt = chains({'letter':"S"})
print(inputt['promptt'], "\n", inputt['answer'])
