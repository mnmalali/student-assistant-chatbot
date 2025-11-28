from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

# splitting the document (OLD DPLITING FUNCTION)
#text_splitter = RecursiveCharacterTextSplitter(
#    chunk_size=300,
#    chunk_overlap=100,
#    length_function=len,
#    is_separator_regex=False,
#)


# Good defaults for short, structured documents
#text_splitter = RecursiveCharacterTextSplitter(
  #  chunk_size=700,          # larger to keep a whole semester or section
   # chunk_overlap=120,       # modest overlap to preserve context across boundaries
  #  separators=[
      #  "\n\n### ", "\n\n## ", "\n\n# ",  # markdown headings, if any
     #   "\n\nYear", "\n\nSemester",        # common section cues in study plans
     #   "\n\n",                            # paragraph
     #   "\n",                              # line
     #   " "                                # fallback
   # ],
   # length_function=len,
   # is_separator_regex=False,
#)




text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,            # big enough for 1 year, small enough not to mix too much
    chunk_overlap=120,         # small overlap so "Year 2" info is still visible
    separators=[
        # 1. Hard university boundaries - highest priority
        r"\n\nNew York University Abu Dhabi",
        r"\n\nKhalifa University \(KU\)",
        r"\n\nAl Ain University \(AAU\)",
        r"\n\nCanadian University Dubai \(CUD\)",
        r"\n\nUniversity of Sharjah",

        # 2. Program level sections (just in case you add more later)
        r"\n\nProgram:",

        # 3. Year and semester boundaries inside each university
        r"\n\nYear\s+\d\s+-",       # "Year 1 - Semester 1 - ..."
        r"\n\nSemester\s+\d\s+-",   # if future versions use separate semester lines

        # 4. Paragraph and line fallbacks
        r"\n\n",
        r"\n",

        # 5. Last resort split on spaces
        r" "
    ],
    length_function=len,
    is_separator_regex=True,
)



# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)