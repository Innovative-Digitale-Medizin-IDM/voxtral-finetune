import os

def get_abs_project_root_path() -> str:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(package_dir, "../../")
    return project_dir