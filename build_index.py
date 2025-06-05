from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import openai, os, textwrap
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

KB_PATH = Path("kb")
INDEX_PATH = Path("index")
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50

embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL"))

texts, metadatas = [], []
for md_file in KB_PATH.glob("**/*.md"):
    content = md_file.read_text(encoding="utf-8")
    # simple sentence‑level splitter
    for i in range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = textwrap.dedent(content[i : i + CHUNK_SIZE])
        texts.append(chunk)
        metadatas.append({"source": str(md_file), "offset": i})

index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
INDEX_PATH.mkdir(exist_ok=True)
index.save_local(folder_path=str(INDEX_PATH))
print("✅  Index built →", INDEX_PATH)