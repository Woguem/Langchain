import operator
from typing import List
from pydantic import BaseModel , Field, ValidationError
from langchain.prompts import PromptTemplate
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from IPython.display import Image, display
from PIL import Image as PILImage
import io




# Load environment variables from .env file
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")



# Initialize the Google Generative AI model
model=ChatGroq(model="gemma2-9b-it")
output=model.invoke("Yo bro")
print(output.content)


# Initialize the Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
len(embeddings.embed_query("Yo bro"))


loader=DirectoryLoader("./data_Prudence", glob="./*.pdf", loader_cls=PyPDFLoader)

documents_rag=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

new_documents_rag=text_splitter.split_documents(documents=documents_rag)

# documents_rag_string=[doc.page_content for doc in new_documents_rag]

print('Document lenght:', len(new_documents_rag))


vectorstore = FAISS.from_documents(new_documents_rag, embedding=embeddings)

# Save the vectorstore to a local directory
vectorstore.save_local("My_Local_Index_FAISS")

retriever=vectorstore.as_retriever(search_kwargs={"k": 3})


# Define the model for the output parser
class TopicSelectionParser(BaseModel):
    Topic:str=Field(description="selected topic")
    Reasoning:str=Field(description='Reasoning behind topic selection') 
    

parser=PydanticOutputParser(pydantic_object=TopicSelectionParser)

print(parser.get_format_instructions())


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define the state graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str
    
    
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

web_crawler_research=TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
    
    

# RAG Function
def rag_function(state:AgentState):
    print("-> RAG Call ->")
    
    question = state["messages"][0]
    
    # Create the RAG prompt
    
    prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are an assistant for question-answering tasks in jury prudence. Use the retrieved information to provide an accurate and concise answer to the question.. If you don't know the answer, just say that you don't know. Strictly follow these formatting instructions :\n\n{format_instructions}"),
    HumanMessagePromptTemplate.from_template("Contexte : {context}\n\nQuestion : {question}")
    ])
    
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(),
        "format_instructions": RunnableLambda(lambda _: parser.get_format_instructions())}
        | prompt
        | model
        | parser      # StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return  {"messages": [result]}




def supervisor_function(state:AgentState):
    
    question=state["messages"][-1]
    
    print("Question",question)
    
    prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Your task is to classify the given user query into one of the following categories: [jurisprudence, not real-time information, not related]. Only respond with the category name and nothing else. Strictly follow these formatting instructions :\n\n{format_instructions}"),
    HumanMessagePromptTemplate.from_template("Question : {question}")
    ])
    
    chain = (
    {
        "question": RunnablePassthrough(),  # goes straight to the question
        "format_instructions": RunnableLambda(lambda _: parser.get_format_instructions())  # injects the setpoint
    }
    | prompt
    | model
    | parser
)
    
    response = chain.invoke({"question":question})
    
    print("Parsed response:", response)
    
    return {"messages": [response.Topic]}



# LLM Function
def llm_function(state:AgentState):
    print("-> LLM Call ->")
    question = state["messages"][0]
    
    # Normal LLM call
    format_instructions = parser.get_format_instructions()

    complete_query = f"""Answer the following question using your knowledge of the real world.
                    Respond ONLY in the JSON format below that matches this schema:

                     {format_instructions}

                    Question: {question}
                    """ 
    chain = model | parser
    response = chain.invoke(complete_query)
    # print(response)
    return {"messages": [response]}


# # Web Crawler Function
# def web_crawler_function(state:AgentState):
#     print("-> Web Crawler Call ->")
#     question = state["messages"][0]
#     chain = web_crawler_research      
#     response = chain.invoke(question)
#     return {"messages": [response]}


def web_crawler_function(state:AgentState):
    print("-> Web Crawler Call ->")
    
    question = state["messages"][0]
    
    # Create the Web Crawler prompt
    
    prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are an assistant for question-answering tasks in jury prudence. Use the retrieved information to provide an accurate and concise answer to the question. If you don't know the answer, just say that you don't know. Strictly follow these formatting instructions :\n\n{format_instructions}"),
    HumanMessagePromptTemplate.from_template("Contexte : {context}\n\nQuestion : {question}")
    ])
    
    # web_crawler_research
    web_crawler_chain = (
        {"context": RunnableLambda(lambda x: web_crawler_research.invoke(question)), 
         "question": RunnablePassthrough(),
        "format_instructions": RunnableLambda(lambda _: parser.get_format_instructions())}
        | prompt
        | model
        | parser      
    )
    result = web_crawler_chain.invoke(question)
    return  {"messages": [result]}




def router_1(state:AgentState):
    print("-> ROUTER 1 ->")
    
    last_message=state["messages"][-1]
    print("last_message:", last_message)
    
    if "jurisprudence" in last_message.lower():
        return "RAG Call"
    elif "not related" in last_message.lower():
        return "LLM Call"
    else:
        return "Web Crawler Call"
    
    
def Validation_function(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    print("-> Validation Call ->")
    
    try:
        if isinstance(last_msg, str):
            parser.parse(last_msg)
        elif isinstance(last_msg, TopicSelectionParser):
            last_msg.model_validate(last_msg)
        else:
            raise ValueError(f"Type de message inattendu : {type(last_msg)}")
    except (ValidationError, ValueError) as e:
        print(f"Validation failed: {e}")
        # redirect to supervisor
        state["next_node"] = "Supervisor"
        return state
    
    # Other validation checks
    if "I don't know" in last_msg.Reasoning.lower() or "I don't know" in last_msg.Topic.lower():
        print("Invalid Output (not informative)")
        state["next_node"] = "Supervisor"
        return state
    
    # Validation OK, End the workflow
    print("Validation successful")
    state["next_node"] = "END"
    return state




# Create the workflow state graph

workflow=StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_function)

workflow.add_node("RAG", rag_function)

workflow.add_node("LLM", llm_function)

workflow.add_node("Web Crawler", web_crawler_function)

workflow.add_node("Validation", Validation_function)


workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    router_1,
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM",
        "Web Crawler Call": "Web Crawler"
    }
)


def router_2(state: AgentState) -> str:
    print("-> ROUTER 2 ->")
    # Returns either ‘Supervisor’ to restart, or ‘END’ to finish.
    result = state["next_node"]
    print("Router 2 result:", result)
    if result == "Supervisor":
        return "Supervisor"
    elif result == "END":
        return "END"
    else:
        raise ValueError(f"Unexpected next_node value: {result}")
    
    


workflow.add_edge("RAG", "Validation")
workflow.add_edge("LLM", "Validation")
workflow.add_edge("Web Crawler", "Validation")


workflow.add_conditional_edges(
    "Validation",
    router_2,
    {
        "Supervisor": "Supervisor",
        "END": END
    }
)

workflow.add_edge("Validation", END)

app=workflow.compile()

# Display the workflow graph in Mermaid format

img_data = app.get_graph().draw_mermaid_png()

image = PILImage.open(io.BytesIO(img_data))

# Save graph image
image.save("graph_output.png")


state={"messages":["Talk me about psg club ?"]}

Result = app.invoke(state)

print("Final Result:", Result)

# Extract Topic and Reasoning
messages = Result.get("messages", [])


print("#"*100)
for msg in messages:
    if isinstance(msg, TopicSelectionParser):
        print("Topic:", msg.Topic)
        print("Reasoning:", msg.Reasoning)
print("#"*100)






























