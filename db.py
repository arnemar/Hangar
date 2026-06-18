import sqlite3

from config import DB_PATH


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            -- ── Chat ──────────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                mode TEXT DEFAULT 'chat',
                model TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                tool_results TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- ── Compiler: file tracking ────────────────────────────────────
            -- Invariant: if hash unchanged, compiler MUST NOT emit IR changes.
            -- Invariant: if last_ir_hash unchanged, graph builder MUST NOT emit node/edge events.
            CREATE TABLE IF NOT EXISTS file_snapshots (
                id               TEXT PRIMARY KEY,
                project_id       TEXT NOT NULL,
                path             TEXT NOT NULL,
                language         TEXT,
                hash             TEXT NOT NULL,
                last_ir_hash     TEXT,
                size             INTEGER,
                last_compiled_at TEXT,
                UNIQUE(project_id, path)
            );

            -- ── Compiler: knowledge graph nodes ───────────────────────────
            -- kind values (physical): project/folder/file/class/interface/enum/
            --   struct/type_alias/function/method/constructor/property/
            --   variable/constant/import/export/namespace/module
            -- kind values (semantic): api_endpoint/route/middleware/db_table/
            --   db_column/migration/config_key/env_var/test_case/docker_service
            -- kind values (document): readme_section/doc_section/api_spec
            CREATE TABLE IF NOT EXISTS nodes (
                id           TEXT PRIMARY KEY,
                project_id   TEXT NOT NULL,
                path         TEXT NOT NULL,
                kind         TEXT NOT NULL,
                name         TEXT NOT NULL,
                parent_id    TEXT,
                signature    TEXT,
                visibility   TEXT DEFAULT 'public',
                language     TEXT,
                start_line   INTEGER,
                end_line     INTEGER,
                source       TEXT,
                summary      TEXT,
                fan_in       INTEGER DEFAULT 0,
                fan_out      INTEGER DEFAULT 0,
                complexity   INTEGER DEFAULT 0,
                importance   REAL    DEFAULT 0.0,
                usage_count  INTEGER DEFAULT 0,
                content_hash TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_project_kind
                ON nodes(project_id, kind);
            CREATE INDEX IF NOT EXISTS idx_nodes_project_path
                ON nodes(project_id, path);
            CREATE INDEX IF NOT EXISTS idx_nodes_project_name
                ON nodes(project_id, name);
            CREATE INDEX IF NOT EXISTS idx_nodes_parent
                ON nodes(parent_id);

            -- FTS5 for exact/prefix symbol search
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                name, signature, summary,
                content=nodes, content_rowid=rowid
            );

            -- FTS5 sync triggers (keep nodes_fts in sync with nodes automatically)
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, name, signature, summary)
                VALUES (new.rowid, new.name, new.signature, new.summary);
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, name, signature, summary)
                VALUES ('delete', old.rowid, old.name, old.signature, old.summary);
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, name, signature, summary)
                VALUES ('delete', old.rowid, old.name, old.signature, old.summary);
                INSERT INTO nodes_fts(rowid, name, signature, summary)
                VALUES (new.rowid, new.name, new.signature, new.summary);
            END;

            -- ── Compiler: knowledge graph edges ───────────────────────────
            -- kind values: contains/calls/imports/inherits/implements/
            --   uses/reads/writes/creates/returns/throws/tests/references
            CREATE TABLE IF NOT EXISTS edges (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                from_id    TEXT NOT NULL,
                to_id      TEXT,
                raw_ref    TEXT,
                kind       TEXT NOT NULL,
                resolved   INTEGER DEFAULT 0,
                confidence REAL    DEFAULT 1.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (from_id) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_from    ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to      ON edges(to_id);
            CREATE INDEX IF NOT EXISTS idx_edges_unresolved
                ON edges(project_id, resolved) WHERE resolved = 0;

            -- ── Compiler: append-only event log ───────────────────────────
            -- type values: FILE_DISCOVERED/FILE_CHANGED/FILE_REMOVED/
            --   NODE_CREATED/NODE_UPDATED/NODE_REMOVED/
            --   EDGE_CREATED/EDGE_RESOLVED/EDGE_REMOVED/
            --   SUMMARY_GENERATED/EMBEDDING_GENERATED/
            --   IMPORTANCE_RECALCULATED/
            --   COMPILATION_STARTED/COMPILATION_COMPLETED/
            --   RETRIEVAL_TRACE (query,intent,retrieved,source_counts,confidence,latency_ms)
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                type       TEXT NOT NULL,
                entity_id  TEXT,
                payload    TEXT,
                timestamp  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_project
                ON events(project_id, timestamp);

            -- ── Compiler: architectural memory ────────────────────────────
            CREATE TABLE IF NOT EXISTS project_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, key)
            );

            -- ── Compiler: hot symbol cache ────────────────────────────────
            CREATE TABLE IF NOT EXISTS hot_symbols (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   TEXT NOT NULL,
                node_id      TEXT NOT NULL,
                session_id   TEXT,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 1,
                UNIQUE(project_id, node_id),
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_hot_symbols_recency
                ON hot_symbols(project_id, last_accessed DESC);

            -- ── Compiler: cached summaries ────────────────────────────────
            CREATE TABLE IF NOT EXISTS summaries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id      TEXT NOT NULL UNIQUE,
                content      TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                model        TEXT,
                created_at   TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            );

            -- ── Compiler: embeddings ──────────────────────────────────────
            CREATE TABLE IF NOT EXISTS embeddings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id      TEXT NOT NULL UNIQUE,
                vector       BLOB NOT NULL,
                model        TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            );

            -- ── Compiler: project compile state ───────────────────────────
            -- Lightweight epoch tracking. is_compiling=1 prevents concurrent
            -- compile runs; graph_version monotonically increments per run.
            -- unresolved_edges is updated by the linker after each compile.
            CREATE TABLE IF NOT EXISTS project_state (
                project_id          TEXT PRIMARY KEY,
                graph_version       INTEGER DEFAULT 0,
                is_compiling        INTEGER DEFAULT 0,
                last_compile_started  TEXT,
                last_compile_finished TEXT,
                total_nodes         INTEGER DEFAULT 0,
                total_edges         INTEGER DEFAULT 0,
                unresolved_edges    INTEGER DEFAULT 0
            );
        """)
        # Sync any nodes that existed before the triggers were created
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        conn.commit()
