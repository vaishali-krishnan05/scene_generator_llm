"""FAISS-backed retrieval for the comic collection."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ComicDocument:
    # Store the main information associated with each comic.
    title: str
    text: str
    json_path: Path
    image_path: Path | None


@dataclass(frozen=True)
class ComicResult:
    # Store a retrieved comic together with its similarity score.
    document: ComicDocument
    score: float


def _story_text(story: dict[str, Any]) -> str:
    # Combine the story style, character descriptions, scenes,
    # and dialogue into one searchable text representation.
    parts = []
    config = story.get("config", {})
    if isinstance(config, dict) and config.get("global_style"):
        parts.append(str(config["global_style"]))

    characters = story.get("characters", {})
    if isinstance(characters, dict):
        parts.extend(
            f"{name}: {description}"
            for name, description in characters.items()
        )

    script = story.get("script", [])
    if isinstance(script, list):
        for panel in script:
            if isinstance(panel, dict):
                parts.extend(
                    str(panel[key])
                    for key in ("scene", "dialogue")
                    if panel.get(key)
                )
    return " ".join(parts).strip()


def load_comic_documents(comics_dir: Path) -> list[ComicDocument]:
    """Load story JSON files, including those awaiting rendered comic images."""
    comics_dir = Path(comics_dir)
    
    # Check that the directory containing the comic collection exists.
    if not comics_dir.is_dir():
        raise FileNotFoundError(f"Comic directory does not exist: {comics_dir}")

    documents = []
    
    # Process every JSON story file in the comic collection.
    for json_path in sorted(comics_dir.glob("*.json")):
        image_path = json_path.with_suffix(".png")
        if not image_path.is_file():
            image_path = None

        text_path = json_path.with_suffix(".txt")
        
        # Use the pre-generated text file when available.
        if text_path.is_file():
            text = text_path.read_text(encoding="utf-8").strip()
        else:
            try:
                # Otherwise, extract searchable text directly from the JSON story.
                story = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Could not read comic story {json_path}: {error}") from error
            
            if not isinstance(story, dict):
                raise ValueError(f"Comic story must contain a JSON object: {json_path}")
            
            text = _story_text(story)

        if not text:
            raise ValueError(f"Comic has no searchable text: {json_path}")
        
        # Create a document object containing the story and optional image.
        documents.append(
            ComicDocument(
                title=json_path.stem.replace("_", " ").title(),
                text=text,
                json_path=json_path,
                image_path=image_path,
            )
        )

    if not documents:
        raise ValueError(
            f"No comic story JSON files were found in {comics_dir}."
        )
    
    return documents


class ComicRAG:
    """Build and query a cosine-similarity FAISS index of comic stories."""

    def __init__(
        self,
        comics_dir: Path,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        try:
            # Import these libraries only when the RAG system is initialized.
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "Comic search requires faiss-cpu, numpy, and sentence-transformers. "
                "Install the project dependencies from requirements.txt."
            ) from error

        self._faiss = faiss
        self._numpy = np
        
        # Load the sentence-transformer model used to create text embeddings.
        self._encoder = SentenceTransformer(model_name)
        
        self.comics_dir = Path(comics_dir)
        self.documents: list[ComicDocument] = []
        self.index: Any = None
        
        # Build the initial searchable FAISS index.
        self.reindex()

    def _encode(self, texts: list[str]):
        # Convert text into numerical embeddings.
        # Normalization allows inner-product search to represent cosine similarity.
        embeddings = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._numpy.asarray(embeddings, dtype=self._numpy.float32)

    def reindex(self) -> int:
        """Rebuild the FAISS index from all story JSON files in the collection."""
        
        # Load all available comic documents before creating the index.
        documents = load_comic_documents(self.comics_dir)
        
        # Convert each document into an embedding vector.
        embeddings = self._encode([document.text for document in documents])
        
        # IndexFlatIP performs inner-product similarity search on the embeddings.
        index = self._faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        self.documents = documents
        self.index = index
        
        return len(documents)

    def search(self, query: str, top_k: int = 3) -> list[ComicResult]:
        # Validate the search query before running the retrieval step.
        if not query or not query.strip():
            raise ValueError("Enter a query to search the comic collection.")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Convert the user's query into the same embedding space as the comics.
        query_embedding = self._encode([query.strip()])
        
        # Retrieve the comics with the highest similarity to the query.
        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.documents)),
        )
        
        # Convert FAISS results into easy-to-use ComicResult objects.
        return [
            ComicResult(
                document=self.documents[int(index)],
                score=float(score),
            )
            for score, index in zip(scores[0], indices[0])
            if 0 <= index < len(self.documents)
        ]