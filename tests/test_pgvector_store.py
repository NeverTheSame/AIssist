"""Integration tests for pgvector_store.py against a real Postgres+pgvector
instance. Skipped unless PGVECTOR_TEST_DSN is set, the same "not run against
a paid/live deployment unless configured" honesty pattern the rest of this
project follows for anything that needs live infrastructure.

Local setup:
    docker run -d --name aissist-pgvector-test -e POSTGRES_PASSWORD=postgres \\
        -p 5544:5432 pgvector/pgvector:pg16
    export PGVECTOR_TEST_DSN=postgresql://postgres:postgres@127.0.0.1:5544/postgres
    pytest tests/test_pgvector_store.py
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PGVECTOR_TEST_DSN"),
    reason="PGVECTOR_TEST_DSN not set; no live Postgres+pgvector to test against",
)


@pytest.fixture
def conn():
    import pgvector_store as pv

    connection = pv.connect(os.environ["PGVECTOR_TEST_DSN"])
    yield connection
    connection.close()


@pytest.fixture
def table():
    # Unique per test run so parallel/repeated runs don't collide.
    return f"test_articles_{uuid.uuid4().hex[:8]}"


class TestSchema:
    def test_ensure_schema_is_idempotent(self, conn, table):
        import pgvector_store as pv

        pv.ensure_schema(conn, table, dimensions=4)
        pv.ensure_schema(conn, table, dimensions=4)  # must not raise
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            assert cur.fetchone()[0] == table
        conn.execute(f'DROP TABLE "{table}"')

    def test_rejects_unsafe_table_name(self, conn):
        import pgvector_store as pv

        with pytest.raises(ValueError):
            pv.ensure_schema(conn, "articles; DROP TABLE articles;--", dimensions=4)


class TestUpsertAndSearch:
    def test_search_returns_nearest_neighbor_first(self, conn, table):
        import pgvector_store as pv

        pv.ensure_schema(conn, table, dimensions=4)
        articles = [
            {"article_path": "apples.md", "title": "Apples", "content_summary": "about apples"},
            {"article_path": "oranges.md", "title": "Oranges", "content_summary": "about oranges"},
            {"article_path": "pears.md", "title": "Pears", "content_summary": "about pears"},
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        written = pv.upsert_articles(conn, table, articles, embeddings)
        assert written == 3

        results = pv.search(conn, table, [0.9, 0.1, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["article_path"] == "apples.md"
        assert results[0]["semantic_similarity"] > results[1]["semantic_similarity"]
        conn.execute(f'DROP TABLE "{table}"')

    def test_upsert_updates_existing_row(self, conn, table):
        import pgvector_store as pv

        pv.ensure_schema(conn, table, dimensions=4)
        pv.upsert_articles(
            conn, table,
            [{"article_path": "a.md", "title": "Old title", "content_summary": "old"}],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        pv.upsert_articles(
            conn, table,
            [{"article_path": "a.md", "title": "New title", "content_summary": "new"}],
            [[0.0, 1.0, 0.0, 0.0]],
        )
        with conn.cursor() as cur:
            cur.execute(f'SELECT title, content_summary FROM "{table}" WHERE article_path = %s', ("a.md",))
            row = cur.fetchone()
        assert row == ("New title", "new")
        conn.execute(f'DROP TABLE "{table}"')

    def test_mismatched_lengths_raise(self, conn, table):
        import pgvector_store as pv

        pv.ensure_schema(conn, table, dimensions=4)
        with pytest.raises(ValueError):
            pv.upsert_articles(
                conn, table,
                [{"article_path": "a.md"}, {"article_path": "b.md"}],
                [[1.0, 0.0, 0.0, 0.0]],
            )
        conn.execute(f'DROP TABLE "{table}"')

    def test_articles_missing_path_are_skipped(self, conn, table):
        import pgvector_store as pv

        pv.ensure_schema(conn, table, dimensions=4)
        written = pv.upsert_articles(
            conn, table,
            [{"title": "no path"}],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        assert written == 0
        conn.execute(f'DROP TABLE "{table}"')
