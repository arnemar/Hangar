"""Register and browse explicit local project folders."""

from starlette.responses import JSONResponse
from starlette.routing import Route

from hangar.api.input import read_object, read_text
from hangar.persistence.storage import get_project, load_projects, save_projects
from hangar.workspace.files import LocalProject, WorkspaceError, register_root


def require_project(name: str) -> LocalProject:
    record = get_project(name)
    if not record:
        raise WorkspaceError("Project not found.", 404)
    return LocalProject(record)


async def list_projects(request):
    return JSONResponse({"projects": [
        {"name": record["name"], "root": record.get("root", ""), "local": not record.get("remote")}
        for record in load_projects() if record.get("name")
    ]})


async def create_project(request):
    data = await read_object(request)
    name = read_text(data.get("name"), "Project name", 100).strip()
    if not name or name in {".", ".."} or any(character in "/\\" or ord(character) < 32 for character in name):
        raise WorkspaceError("Enter a project name without slashes.")
    records = load_projects()
    if any(record.get("name") == name for record in records):
        raise WorkspaceError("A project already uses that name.", 409)
    root = str(register_root(read_text(data.get("root"), "Folder path", 32767)))
    save_projects([*records, {"name": name, "root": root, "scope_version": 1}])
    return JSONResponse({"name": name, "root": root, "local": True}, status_code=201)


async def list_files(request):
    project = require_project(request.path_params["name"])
    return JSONResponse(project.entries(request.query_params.get("path", "")))


async def read_file(request):
    path = request.query_params.get("path")
    if not path:
        raise WorkspaceError("Enter a file path.")
    project = require_project(request.path_params["name"])
    return JSONResponse({"path": path, "content": project.read(path)})


routes = [
    Route("/api/projects", list_projects),
    Route("/api/projects", create_project, methods=["POST"]),
    Route("/api/projects/{name}/files", list_files),
    Route("/api/projects/{name}/file", read_file),
]
