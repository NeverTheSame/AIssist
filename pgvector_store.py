"""pgvector-backed retrieval for knowledge-base articles.

An alternative to the Qdrant path in ``article_searcher.py``, kept to the
same embedding space (all-MiniLM-L6-v2, 384 dimensions) so a table built
here is a drop-in swap for a Qdrant collection: ``search()`` returns
candidates shaped exactly like the Qdrant branch of
``ArticleSearcher._semantic_search`` -- ``article_path``, ``title``,
``content_summary``, ``semantic_similarity``.

Selected with ``PGVECTOR_DSN`` (see config.py / .env.example). Nothing here
runs unless that's set, so behavior for existing Qdrant/JSON deployments is
unchanged.
"""

import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 384


def connect(dsn: str):
    """Open a connection with the pgvector type adapter registered.

    Raises ImportError with a install hint if psycopg/pgvector aren't present.
    """
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise ImportError(
            "pgvector support requires the 'psycopg[binary]' and 'pgvector' "
            "packages: pip install 'psycopg[binary]' pgvector"
        ) from exc

    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn, table: str, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
    """Create the article table and its ANN index if they don't already exist."""
    name = _quote(table)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {name} (
            article_path TEXT PRIMARY KEY,
            title TEXT,
            content_summary TEXT,
            embedding VECTOR({dimensions})
        )
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {_quote(table + '_embedding_idx')}
        ON {name} USING hnsw (embedding vector_cosine_ops)
        """
    )


def upsert_articles(
    conn,
    table: str,
    articles: Sequence[Dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """Insert or update article rows keyed on article_path. Returns rows written."""
    if len(articles) != len(embeddings):
        raise ValueError(
            f"articles ({len(articles)}) and embeddings ({len(embeddings)}) "
            "must be the same length"
        )
    rows = []
    for article, embedding in zip(articles, embeddings):
        path = article.get("article_path") or article.get("url") or article.get("path")
        if not path:
            continue
        summary = article.get("content_summary") or (article.get("content") or "")[:600]
        rows.append((path, article.get("title", ""), summary, list(embedding)))

    if not rows:
        return 0

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {_quote(table)} (article_path, title, content_summary, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (article_path) DO UPDATE SET
                title = EXCLUDED.title,
                content_summary = EXCLUDED.content_summary,
                embedding = EXCLUDED.embedding
            """,
            rows,
        )
    return len(rows)


def search(
    conn, table: str, query_embedding: Sequence[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Cosine-similarity nearest-neighbor search.

    Returns candidates shaped like the Qdrant path so callers in
    ArticleSearcher don't need to branch on which backend answered.
    """
    vec = list(query_embedding)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT article_path, title, content_summary,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {_quote(table)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "article_path": row[0],
            "title": row[1],
            "content_summary": row[2],
            "semantic_similarity": float(row[3]),
        }
        for row in rows
    ]


def _quote(identifier: str) -> str:
    """Double-quote a SQL identifier after checking it's a plain name.

    Table names here come from config (PGVECTOR_TABLE), not request input,
    but an identifier still shouldn't be string-formatted into SQL unchecked.
    """
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"unsafe identifier: {identifier!r}")
    return f'"{identifier}"'
