#import necessary libraries
from datetime import datetime, timezone
import email
from email import policy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup
import chromadb
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

#intialize GROQ client
load_dotenv()
groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
groq_client: Optional[Groq] = Groq(api_key=groq_api_key) if groq_api_key else None

#parsing document
def parse_document(file_path: Union[str, Path]) -> str:
    """Parse documents of various formats (PDF, HTML, EML, TXT, JSON, etc.) and extract text.

    Args:
        file_path: Path to the target document.

    Returns:
        Extracted plain text content.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix: str = path.suffix.lower()

    try:
        if suffix == ".pdf":
            reader = pypdf.PdfReader(str(path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(text.strip() for text in pages_text if text.strip())

        elif suffix in (".docx", ".doc"):
            import docx
            doc = docx.Document(str(path))
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paras.append(row_text)
            return "\n\n".join(paras)

        elif suffix in (".html", ".htm"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                for element in soup(["script", "style", "noscript"]):
                    element.decompose()
                return soup.get_text(separator="\n", strip=True)

        elif suffix == ".eml":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                msg = email.message_from_file(f, policy=policy.default)
                headers = []
                for header_key in ("From", "To", "Subject", "Date", "Message-ID"):
                    val = msg.get(header_key)
                    if val:
                        headers.append(f"{header_key}: {val}")
                header_text = "\n".join(headers)

                body_text = ""
                body_part = msg.get_body(preferencelist=("plain", "html"))
                if body_part:
                    raw_body = body_part.get_content()
                    if body_part.get_content_type() == "text/html":
                        soup = BeautifulSoup(raw_body, "html.parser")
                        body_text = soup.get_text(separator="\n", strip=True)
                    else:
                        body_text = str(raw_body).strip()

                return f"{header_text}\n\n{body_text}".strip()

        elif suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                return json.dumps(data, indent=2)

        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().strip()

    except Exception as e:
        print(f"Error parsing document '{file_path}': {e}")
        return ""

#extract and genrate metadata like docID, creation date , modified dates ,version etc also genrate ai metadata for semantic retrival
def extract_and_generate_metadata(
    file_path: Union[str, Path],
    content: Optional[str] = None,
    use_ai: bool = True
) -> Dict[str, Any]:
    """Extract standard metadata (docID, dates, version) and generate AI metadata for semantic retrieval.

    Args:
        file_path: Path to the target document.
        content: Optional extracted text content. If None, the document is parsed.
        use_ai: Flag to enable AI-powered metadata generation via Groq.

    Returns:
        Dictionary containing document metadata and semantic AI metadata.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if content is None:
        content = parse_document(path)

    # 1. System and document metadata
    stat = path.stat()
    created_dt = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
    modified_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    version_match = re.search(
        r"(?:version|v)[\s:]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
        content,
        re.IGNORECASE
    )
    version = version_match.group(1) if version_match else "1.0"

    doc_id_match = re.search(r"([A-Z]{2,4}-\d+)", path.name)
    doc_id = doc_id_match.group(1) if doc_id_match else path.stem

    metadata: Dict[str, Any] = {
        "doc_id": doc_id,
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_type": path.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "creation_date": created_dt,
        "modified_date": modified_dt,
        "version": version,
        "content_hash": hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest(),
        "ai_metadata": {}
    }

    # 2. AI metadata for semantic retrieval
    if use_ai and groq_client and content.strip():
        content_preview = content[:4000]
        prompt = (
            "Analyze the following enterprise document and extract structured metadata for semantic retrieval.\n"
            "Return ONLY a valid JSON object with the following keys:\n"
            "- summary: A concise summary of the document (2-3 sentences).\n"
            "- category: Primary category (e.g., Troubleshooting, Policy, Incident Report, Architecture, Communication).\n"
            "- keywords: A list of 5-8 salient technical keywords.\n"
            "- entities: Specific enterprise entities mentioned (e.g., Asset IDs like AST001, Employee IDs like EMP001, Incident IDs like INC01001, system names).\n"
            "- semantic_queries: 3-5 representative user queries or questions this document can answer.\n\n"
            f"Document Content:\n{content_preview}"
        )

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an enterprise knowledge extraction assistant. Respond with strictly valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            ai_output = response.choices[0].message.content
            if ai_output:
                metadata["ai_metadata"] = json.loads(ai_output)
        except Exception as e:
            metadata["ai_metadata"] = {
                "summary": content[:200].strip(),
                "category": "General",
                "keywords": [],
                "entities": [],
                "semantic_queries": [],
                "error": str(e)
            }
    else:
        metadata["ai_metadata"] = {
            "summary": content[:200].strip() if content else "",
            "category": "General",
            "keywords": [],
            "entities": [],
            "semantic_queries": []
        }

    return metadata

#recursive character chunking
def chunk_document(
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[str]:
    """Split text content into chunks using recursive character splitting.

    Args:
        content: Text content to split.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        List of text chunks.
    """
    if not content or not content.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(content)

#initialize hugging face embedding and chromaDB client
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"local_files_only": True} if os.environ.get("HF_HUB_OFFLINE") == "1" else {}
)

CHROMA_PERSIST_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "chroma_db")
)
chroma_client: chromadb.PersistentClient = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
knowledge_collection: chromadb.Collection = chroma_client.get_or_create_collection(
    name="enterprise_knowledge",
    metadata={"hnsw:space": "cosine"}
)

#embed and store knowledge_base documents and metadata in chromaDB
def sanitize_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """Sanitize metadata values into primitives compatible with ChromaDB.

    Args:
        metadata: Nested raw metadata dictionary.

    Returns:
        Flat dictionary containing only primitive types (str, int, float, bool).
    """
    clean_meta: Dict[str, Union[str, int, float, bool]] = {
        "doc_id": str(metadata.get("doc_id", "")),
        "file_name": str(metadata.get("file_name", "")),
        "file_path": str(metadata.get("file_path", "")),
        "file_type": str(metadata.get("file_type", "")),
        "file_size_bytes": int(metadata.get("file_size_bytes", 0)),
        "creation_date": str(metadata.get("creation_date", "")),
        "modified_date": str(metadata.get("modified_date", "")),
        "version": str(metadata.get("version", "1.0")),
    }

    ai_meta = metadata.get("ai_metadata", {})
    if isinstance(ai_meta, dict):
        clean_meta["summary"] = str(ai_meta.get("summary", ""))
        clean_meta["category"] = str(ai_meta.get("category", "General"))

        keywords = ai_meta.get("keywords", [])
        clean_meta["keywords"] = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)

        entities = ai_meta.get("entities", [])
        clean_meta["entities"] = ", ".join(entities) if isinstance(entities, list) else str(entities)

        queries = ai_meta.get("semantic_queries", [])
        clean_meta["semantic_queries"] = " | ".join(queries) if isinstance(queries, list) else str(queries)

    return clean_meta


def store_document_in_chromadb(
    file_path: Union[str, Path],
    collection: Optional[chromadb.Collection] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    use_ai_metadata: bool = True
) -> int:
    """Parse, chunk, extract metadata, generate embeddings, and store in ChromaDB.

    Args:
        file_path: Path to the target document.
        collection: Target ChromaDB collection. Defaults to knowledge_collection.
        chunk_size: Size for recursive character chunking.
        chunk_overlap: Overlap for recursive character chunking.
        use_ai_metadata: Whether to generate AI metadata with Groq.

    Returns:
        Number of chunks stored.
    """
    target_collection = collection or knowledge_collection
    path = Path(file_path)

    # 1. Parse and extract metadata
    content = parse_document(path)
    if not content.strip():
        return 0

    metadata = extract_and_generate_metadata(path, content=content, use_ai=use_ai_metadata)
    base_metadata = sanitize_metadata_for_chroma(metadata)

    # 2. Chunk document
    chunks = chunk_document(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        chunks = [content]

    # 3. Prepare ChromaDB payloads
    doc_id = base_metadata["doc_id"]
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Union[str, int, float, bool]]] = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx}"
        chunk_metadata = dict(base_metadata)
        chunk_metadata["chunk_index"] = idx
        chunk_metadata["total_chunks"] = len(chunks)

        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(chunk_metadata)

    # Also store a dedicated document-level metadata record for semantic lookup
    if base_metadata.get("summary"):
        summary_id = f"{doc_id}_summary"
        summary_text = (
            f"Document ID: {doc_id}\n"
            f"Title: {path.name}\n"
            f"Category: {base_metadata.get('category', 'General')}\n"
            f"Summary: {base_metadata.get('summary', '')}\n"
            f"Keywords: {base_metadata.get('keywords', '')}\n"
            f"Entities: {base_metadata.get('entities', '')}"
        )
        summary_metadata = dict(base_metadata)
        summary_metadata["is_summary_record"] = True
        summary_metadata["chunk_index"] = -1
        summary_metadata["total_chunks"] = len(chunks)

        ids.append(summary_id)
        documents.append(summary_text)
        metadatas.append(summary_metadata)

    # 4. Generate embeddings with Hugging Face and add to ChromaDB
    embeddings = hf_embeddings.embed_documents(documents)
    target_collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(documents)


def ingest_knowledge_base(
    knowledge_base_dir: Union[str, Path] = "knowledge_base",
    collection: Optional[chromadb.Collection] = None,
    use_ai_metadata: bool = False
) -> Dict[str, int]:
    """Ingest all unstructured and structured documents from the knowledge base directory into ChromaDB.

    Args:
        knowledge_base_dir: Base directory path for knowledge base.
        collection: Target ChromaDB collection.
        use_ai_metadata: Whether to generate AI metadata via Groq during batch ingestion.

    Returns:
        Summary dictionary with counts of files processed and chunks stored.
    """
    kb_path = Path(knowledge_base_dir)
    if not kb_path.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {knowledge_base_dir}")

    supported_suffixes = {".pdf", ".html", ".htm", ".eml", ".txt", ".json", ".md"}
    total_files = 0
    total_chunks = 0

    for file_path in kb_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes:
            # Skip SQLite database binary file if traversed
            if file_path.suffix.lower() in (".sqlite", ".db"):
                continue
            try:
                stored = store_document_in_chromadb(
                    file_path=file_path,
                    collection=collection,
                    use_ai_metadata=use_ai_metadata
                )
                total_files += 1
                total_chunks += stored
            except Exception as e:
                print(f"Error ingesting {file_path}: {e}")

    return {
        "files_ingested": total_files,
        "total_records_stored": total_chunks
    }
