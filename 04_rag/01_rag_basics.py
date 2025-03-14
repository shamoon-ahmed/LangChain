from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma

API_KEY = "AIzaSyAgRy8gTnQSGioorDXy-ZJSATvbk_34ynU"

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                             temperature=0.7,
                             api_key=API_KEY)

loader = YoutubeLoader.from_youtube_url('https://youtu.be/wd7TZ4w1mSw?si=KO2SYIk4P41enDr2')
transcript = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.split_documents(transcript)

embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',
                                          google_api_key=API_KEY)

print("heyyyyy")

print(docs[0])

# vectorstore = Chroma.from_documents(docs, embedding=embeddings)
# retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':10})

# result = retriever.invoke('langchain')
# print(result)