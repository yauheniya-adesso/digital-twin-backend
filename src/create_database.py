"""
Digital Twin - Database Creation
Loads documents and creates vector store
"""

import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def load_documents():
    """Load all documents from data_processed folder."""
    data_dir = Path("data_processed")
    documents = {
        "linkedin": [],
        "github": [],
        "medium": []
    }
    
    # Load LinkedIn JSON
    linkedin_path = data_dir / "linkedin.json"
    if linkedin_path.exists():
        with open(linkedin_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            text = json.dumps(data, indent=2)
            documents["linkedin"].append({
                "content": text,
                "source": "linkedin",
                "metadata": {"type": "profile"}
            })
    
    # Load GitHub READMEs
    github_folder = data_dir / "github_readmes"
    if github_folder.exists():
        for readme in github_folder.glob("*.md"):
            with open(readme, "r", encoding="utf-8") as f:
                documents["github"].append({
                    "content": f.read(),
                    "source": "github",
                    "metadata": {"file": readme.name, "type": "project"}
                })
    
    # Load GitHub Work READMEs
    github_folder = data_dir / "github_work_readmes"
    if github_folder.exists():
        for readme in github_folder.glob("*.md"):
            with open(readme, "r", encoding="utf-8") as f:
                documents["github"].append({
                    "content": f.read(),
                    "source": "github",
                    "metadata": {"file": readme.name, "type": "project"}
                })
    
    # Load Medium articles
    medium_folder = data_dir / "medium_articles"
    if medium_folder.exists():
        for article in medium_folder.glob("*.md"):
            with open(article, "r", encoding="utf-8") as f:
                documents["medium"].append({
                    "content": f.read(),
                    "source": "medium",
                    "metadata": {"file": article.name, "type": "article"}
                })
    
    return documents


def create_vector_store(force_recreate=False):
    """Create unified vector store with all documents."""
    persist_dir = Path("chroma_db")
    
    # Check if vector store already exists
    if not force_recreate and persist_dir.exists() and any(persist_dir.iterdir()):
        print("⚠ Vector store already exists!")
        print(f"Location: {persist_dir}")
        print("Use --force flag to recreate it.")
        return
    
    print("Creating new vector store...")
    documents = load_documents()
    
    # Debug: show what was loaded
    print(f"\nLoaded documents by source:")
    for source_type, docs in documents.items():
        print(f"  {source_type}: {len(docs)} documents")
    
    # No chunking - each file is a single chunk
    all_texts = []
    all_metadatas = []
    
    for source_type, docs in documents.items():
        for doc in docs:
            all_texts.append(doc["content"])
            all_metadatas.append({
                "source": source_type,
                **doc["metadata"]
            })
    
    print(f"Processing {len(all_texts)} documents (1 document = 1 chunk)...")
    
    # Create embeddings 
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    
    # Create vector store with persistence
    vector_store = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        collection_name="digital_twin",
        persist_directory=str(persist_dir)
    )
    
    print(f"✓ Created and saved {len(all_texts)} chunks")
    print(f"✓ Vector store persisted to {persist_dir}")
    print("\nNow you can run: python -m src.run_agent")


if __name__ == "__main__":
    import sys
    
    force = "--force" in sys.argv or "-f" in sys.argv
    
    if force:
        print("Force recreate mode enabled\n")
    
    create_vector_store(force_recreate=force)