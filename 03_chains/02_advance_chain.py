from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                             api_key=os.getenv('GEMINI_API_KEY'),
                             temperature=0.7)

loader = YoutubeLoader.from_youtube_url('https://youtu.be/ZXiruGOCn9s?si=2VF8GhjyplU1imsK', add_video_info=False)
docs = loader.load()
yt_transcript = docs[0].page_content

prompt_temp = 'Analyze the given youtube transcript: {transcript}, generate a short summary'
question_temp = 'What libraries or frameworks did the speaker talk about in the {summary}?'

prompt = PromptTemplate.from_template(prompt_temp)
chain_1 = LLMChain(llm=llm, prompt=prompt, output_key='summary')

quest = PromptTemplate.from_template(question_temp)
chain_2 = LLMChain(llm=llm, prompt=quest, output_key='answer')

yt_chains = SequentialChain(chains=[chain_1, chain_2],
                            input_variables=['transcript'],
                            output_variables=['summary', 'answer'], verbose=True)
outputt = yt_chains({'transcript':yt_transcript})
print("Summary: ",outputt['summary'], "\n", "Answer: ", outputt['answer'])

print("\n", "-------------------------------------", "\n")

promptt = PromptTemplate(template=prompt_temp, input_variables=['transcript'])
llm_chain = promptt | llm
summ = llm_chain.invoke({'transcript': yt_transcript}).content
print(summ)