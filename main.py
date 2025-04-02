### RAG using docling, qdrant, fastembed and deepseek
#%%
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

converter = DocumentConverter()

#%%
# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------

sitemap_urls = get_sitemap_urls(base_url="https://www.deductibles.be/",sitemap_filename="sitemap-en.xml")
#remove several urls that are not needed 
remove_urls = ["https://www.deductibles.be" , "https://www.deductibles.be/[slug]","https://www.deductibles.be/all"] 
 
sitemap_urls = [url for url in sitemap_urls if url not in remove_urls]
conv_results_iter = converter.convert_all(sitemap_urls)

docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)

#%% Export to markdown
docs_markdown =[doc.export_to_markdown() for doc in docs]

#%% 
# --------------------------------------------------------------
# Chunking the data
# --------------------------------------------------------------
from docling.chunking import HybridChunker
from itertools import chain

chunker = HybridChunker(
    merge_peers=True,
)

# Merge all iterators into one
chunk_iter = [chunker.chunk(dl_doc=doc) for doc in docs] 
chunk_iter = chain(*chunk_iter)
chunks = list(chunk_iter)

# Analyzing the model_dump we see there is a lot of information that we can use to create a table

# --------------------------------------------------------------
# Embedding
# --------------------------------------------------------------

# Initialize the client
#%%

# in this example we will use the Qdrant (uses FastEmbed to generate vector emmbedings)
 
from qdrant_client import QdrantClient
client = QdrantClient(":memory:")  # Qdrant is running from RAM.

# The :memory: mode is a Python imitation of Qdrant's APIs for prototyping and CI.
# For production deployments, use the Docker image: docker run -p 6333:6333 qdrant/qdrant
# client = QdrantClient(location="http://localhost:6333")

client.set_model("sentence-transformers/all-MiniLM-L6-v2")
client.set_sparse_model("Qdrant/bm25")

# Key Differences:
# Dense Models (e.g., Sentence Transformers):

# Use dense vectors to represent text, capturing complex semantic relationships.
# Suitable for tasks that require understanding the meaning and context of text.
# Often used in neural network-based approaches.
# Sparse Models (e.g., BM25):

# Use sparse vectors, where only a few elements are non-zero, typically representing the presence or frequency of terms.
# Suitable for traditional information retrieval tasks, where the focus is on matching query terms to document terms.
# Often used in keyword-based search systems.
# Use Case:
# In a hybrid search system, you might use both dense and sparse models to leverage their strengths. The dense model can capture semantic similarity, while the sparse model can efficiently handle exact term matching, improving overall search performance and relevance.

# create the documents 
# %%
documents, metadatas = [], []

for chunk in chunks:
    documents.append(chunk.text)
    metadatas.append(
        {
            "filename": chunk.meta.origin.filename,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        }
    )


# Upload the documents to Qdrant   
#%% 
_ = client.add(
    collection_name="accounting_data_3",
    documents=documents,
    metadata=metadatas,
    ids = range(len(documents)),
    batch_size=64,
)

#%%
# uncomment just in case you want to see the embedding of your query
# from fastembed import TextEmbedding

# dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
# query_text  = "can I deduct accounting fees ?"
# query_embedded = list(dense_embedding_model.embed(query_text))[0]

#%%
points = client.query(
    collection_name="accounting_data_3",
    query_text="what is the deductibility rate of cars ?",
    limit=5,
    with_vector = True # only set to True to retrieve the vectors
)

#%%
for i, point in enumerate(points):
    print(f"=== {i} ===")
    print(f"=== {point.score} ===")
    print(point.metadata['title'])
    print(point.document)

    print()


# %%
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(os.getcwd() + '/.env')
client_open_ai = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

def query_open_ai(prompt):
    
    completion = client_open_ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
    ]
    )

    return completion.choices[0].message
    
#%%

# Make a general function to make querying easier 
def rag(question: str, n_points: int = 3) -> str:
    results = client.query(
        collection_name="accounting_data_3",
        query_text=question,
        limit=n_points,
    )

    context = "\n".join(r.document for r in results)
 
    metaprompt = f"""
    You are a senior accountant in Belgium. 
    Answer the following question using the provided context. 
    
    Question: {question.strip()}
    
    Context: 
    {context.strip()}
    
    Answer:
    """

    return query_open_ai(metaprompt)

#%%
temp = rag("are car related expenses deductable ? provide it as a table with rate of deductibility for taxes and VAT", n_points = 1) 

print(temp.content)
# %%
# output the data to keep it in store in case the source website ever changes 
# Ensure the data directory exists
# os.makedirs("data", exist_ok=True)

# # Write the markdown data to a file
# with open("data/output.md", "w", encoding="utf-8") as f:
#     for doc in docs_markdown:
#         f.write(doc + "\n\n")


#%%

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------

# pdf_path = "C:/Users/vince/OneDrive/Bureau/AI/RAG/docling/ai-cookbook/inputs/charges-deductibles-brochure-202409.pdf"
# result = converter.convert(pdf_path)

# document = result.document
# markdown_output = document.export_to_markdown()
# json_output = document.export_to_dict()

# print(markdown_output)