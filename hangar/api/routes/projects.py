"""Register and browse explicit local project folders."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from hangar.persistence.storage import get_project, load_projects, save_projects
from hangar.workspace.files import LocalProject, WorkspaceError, register_root

router = APIRouter()


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    root: str = Field(min_length=1)


def require_project(name: str) -> LocalProject:
    record = get_project(name)
    if not record:
        raise WorkspaceError("Project not found.", 404)
    return LocalProject(record)


@router.get("/api/projects")
async def get_projects():
    return {
        "projects": [
            {"name": record["name"], "root": record.get("root", ""), "local": not record.get("remote")}
            for record in load_projects()
            if isinstance(record, dict) and record.get("name")
        ],
    }


@router.post("/api/projects", status_code=201)
async def create_project(request: ProjectCreate):
    name = request.name.strip()
    if not name or any(character in name for character in "/\\"):
        raise WorkspaceError("Enter a project name without slashes.")
    records = load_projects()
    if any(record.get("name") == name for record in records):
        raise WorkspaceError("A project already uses that name.", 409)
    root = str(register_root(request.root))
    save_projects([*records, {"name": name, "root": root, "scope_version": 1}])
    return {"name": name, "root": root, "local": True}


@router.get("/api/projects/{name}/files")
async def list_files(name: str, path: str = ""):
    return require_project(name).entries(path)


@router.get("/api/projects/{name}/file")
async def read_file(name: str, path: str):
    return {"path": path, "content": require_project(name).read(path)}
