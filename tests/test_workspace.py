"""Regressions for project access and reviewed edits."""

import sqlite3
from contextlib import closing

import pytest

from hangar.persistence import conversations, db, storage
from hangar.workspace import files
from hangar.workspace.changes import changes_for, decide_change, propose_change
from hangar.workspace.files import LocalProject, WorkspaceError
from hangar.workspace.tools import run_tool, schemas


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "workspace.db")
    monkeypatch.setattr(storage, "PROJECTS_PATH", tmp_path / "projects.json")
    monkeypatch.setattr(files, "DATA_DIR", tmp_path / "runtime")
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_bytes(b"first\r\nsecond\r\n")
    record = {"name": "Example", "root": str(root.resolve()), "scope_version": 1}
    storage.save_projects([record])
    db.init_chat_db()
    conversations.migrate_workspace()
    session = conversations.create_session("test-model", record["name"])
    return LocalProject(record), session["id"]


def test_file_list_and_read_stay_inside_attached_folder(workspace, tmp_path):
    project, _ = workspace
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    (project.root / ".env").write_text("private", encoding="utf-8")
    assert [item["name"] for item in project.entries()["entries"]] == ["notes.txt"]
    for unavailable in ("../outside.txt", str(tmp_path / "outside.txt"), ".env"):
        with pytest.raises(WorkspaceError):
            project.read(unavailable)
    assert project.read("notes.txt") == "first\r\nsecond\r\n"


def test_links_do_not_grant_more_file_access(workspace, tmp_path):
    project, _ = workspace
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = project.root / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("This OS account cannot create symbolic links")
    with pytest.raises(WorkspaceError):
        project.read("link.txt")
    assert "link.txt" not in {entry["name"] for entry in project.entries()["entries"]}


def test_model_cannot_apply_or_propose_without_permission(workspace):
    project, session = workspace
    names = {schema["function"]["name"] for schema in schemas(False)}
    assert names == {"list_project_files", "read_project_file"}
    for name in ("apply_file_change", "propose_file_change"):
        output, proposal = run_tool(name, {}, session, project, False)
        assert proposal is None and "unavailable" in output
    assert changes_for(session) == []


def test_proposals_keep_files_unchanged_until_approval(workspace):
    project, session = workspace
    change = propose_change(session, project, "notes.txt", "replace", "third\nfourth", "second")
    assert change["status"] == "pending" and "+third" in change["diff"]
    assert project.read("notes.txt") == "first\r\nsecond\r\n"
    assert decide_change(session, change["id"], "apply")["status"] == "applied"
    assert project.read("notes.txt") == "first\r\nthird\r\nfourth\r\n"
    assert decide_change(session, change["id"], "apply")["status"] == "applied"


def test_stale_proposal_cannot_overwrite_a_local_edit(workspace):
    project, session = workspace
    change = propose_change(session, project, "notes.txt", "replace", "third", "second")
    (project.root / "notes.txt").write_text("user edited this", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="changed since") as error:
        decide_change(session, change["id"], "apply")
    assert error.value.status == 409
    assert project.read("notes.txt") == "user edited this"
    assert changes_for(session)[0]["status"] == "pending"
    assert decide_change(session, change["id"], "dismiss")["status"] == "dismissed"


def test_new_file_requires_the_original_attached_project(workspace):
    project, session = workspace
    change = propose_change(session, project, "new.txt", "create", "new\n")
    assert not (project.root / "new.txt").exists()
    conversations.update_session(session, {"project_name": None})
    with pytest.raises(WorkspaceError, match="Attach the original project"):
        decide_change(session, change["id"], "apply")
    conversations.update_session(session, {"project_name": project.name})
    assert decide_change(session, change["id"], "apply")["status"] == "applied"
    assert project.read("new.txt") == "new\n"


def test_migration_preserves_legacy_sessions_and_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "old.db")
    db.init_chat_db()
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        conn.execute("INSERT INTO sessions VALUES ('s','Old chat','agent','old-model','1','1')")
        conn.execute("INSERT INTO messages VALUES ('m','s','user','Hello',NULL,NULL,'1')")
    conversations.migrate_workspace()
    conversations.migrate_workspace()
    assert conversations.list_sessions()[0]["title"] == "Old chat"
    message = conversations.messages_for("s")[0]
    assert message["content"] == "Hello" and message["attachments"] == []


def test_chat_storage_initializes_without_the_compiler(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "chat.db")
    db.init_chat_db()
    conversations.migrate_workspace()
    with db.get_db() as conn:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "messages", "file_changes"} <= tables
    assert "nodes_fts" not in tables
