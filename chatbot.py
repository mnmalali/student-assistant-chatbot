from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model
llm = ChatOpenAI(temperature=0.4, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)




# Student Assistant system prompt
SYSTEM_PROMPT = """
You are Student Assistant, a helpful and knowledgeable chatbot designed to answer questions about
Computer Science courses, curricula, study plans, prerequisites, and degree structures.

Your goal is to provide clear, accurate, and easy-to-understand answers based strictly on the
information given in “The knowledge” section.

Follow these rules:

1. Use only the information provided in The knowledge section.
   - Do not rely on outside knowledge.
   - If something is missing or not mentioned, clearly say: 
     “This information is not specified in the study plan.”

2. Present answers in a clean and readable format:
   - Use bullet points for lists  
   - Use short paragraphs  
   - Include course codes, names, and credits whenever available  
   - Group items logically (for example, by semester or category)

3. Be concise but helpful.
   - Avoid long, complicated explanations.
   - Focus on exactly what the student needs.

4. Never invent:
   - New courses
   - New prerequisites
   - University rules or policies not in the knowledge

5. When users ask general questions (for example: “What should I take next?”):
   - Base your answer only on what the document says.
   - Avoid giving personal or academic advice outside the provided knowledge.

Your tone:
- Friendly
- Clear
- Professional
- Student-focused

Your role:
Help students understand their course plans quickly and easily using only verified information from the retrieved data.
"""





# Set up the vectorstore to be the retriever (OLD retreiver)
#num_results = 5
#retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

#(NEW Retriever)
num_results = 5
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": num_results, "fetch_k": 12, "lambda_mult": 0.5},
)





#(NEW STREAM RESPONCE)
# call this function for every message added to the chatbot
def stream_response(message, history):
    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # make the call to the LLM (including prompt)
    if message is not None:
        partial_message = ""

        rag_prompt = f"""
        {SYSTEM_PROMPT}

        User question:
        {message}

        Conversation history:
        {history}

        The knowledge:
        {knowledge}
         """

        # Uncomment this if you want to see the full prompt in the terminal for debugging
        # print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message








# initiate the Gradio app
#chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    #container=False,
    #autoscroll=True,
    #scale=7),
#)


# initiate the (NEW) Gradio app
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Ask about CS courses, semesters, credits, or prerequisites...",
        container=False,
        autoscroll=True,
        scale=7,
    ),
    title="Student Assistant – CS Course Plans (UoS, CUD, AAU, KU, NYUAD)",
    description="Ask questions about Computer Science study plans and curricula for UoS, CUD, AAU, KU, and NYUAD.",
)

# launch the Gradio app
chatbot.launch()