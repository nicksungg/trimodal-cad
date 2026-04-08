import json
import subprocess
import trimesh
from trimesh.sample import sample_surface
import numpy as np
from plyfile import PlyData, PlyElement
import os
import random
from scipy.spatial import cKDTree as KDTree
import ast
import re
import sys

def read_jsonl(file_path, *keys):
    """
    Reads a JSONL file and extracts specific keys from each dictionary.

    Args:
        file_path (str): Path to the JSONL file.
        *keys (str): One or more keys to extract from each JSON object.

    Returns:
        tuple: A tuple of lists, each corresponding to the extracted values for a given key.
    """
    results = {key: [] for key in keys}  # Create a dictionary to store lists for each key

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # Parse each line as a JSON object
            for key in keys:
                results[key].append(data.get(key, None))  # Append value or None if key is missing

    return tuple(results[key] for key in keys)  # Return lists as a tuple


def write_python_file(file_content, py_path):
    with open(py_path, "w", encoding="utf-8") as file:
        file.write(file_content)
    return

def run_python_script(py_path):
    try:
        result = subprocess.run(
            [sys.executable, py_path],
            capture_output=True,      # Capture stdout and stderr
            text=True,                # Decode output as text
            check=True                # Raise an exception if the script fails
        )
        return True
    except subprocess.CalledProcessError as e:
        print("error:", e)
        return False


def clean_code_fences(code: str) -> str:
    if "```" not in code:
        return code
    return re.sub(r"```[a-zA-Z]*\n|```", "", code)


def is_feasible_cad_script(code: str, code_language: str):
    code = clean_code_fences(code).strip()
    if not code:
        return False, "empty_script", code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "syntax_error", code

    banned_modules = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "shutil",
        "pathlib",
        "multiprocessing",
        "threading",
    }
    banned_calls = {"exec", "eval", "compile", "open", "__import__"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_name = alias.name.split(".")[0]
                if top_name in banned_modules:
                    return False, f"banned_import:{top_name}", code
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_name = node.module.split(".")[0]
                if top_name in banned_modules:
                    return False, f"banned_import:{top_name}", code
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in banned_calls:
                return False, f"banned_call:{node.func.id}", code

    if code_language.lower() == "cadquery":
        has_cadquery_signal = ("cq." in code) or ("cadquery" in code)
        has_model_ops = any(
            token in code
            for token in [
                ".box(",
                ".extrude(",
                ".revolve(",
                ".loft(",
                ".sweep(",
                ".sphere(",
                ".cylinder(",
                ".union(",
                ".cut(",
                ".intersect(",
            ]
        )
        if not has_cadquery_signal:
            return False, "missing_cadquery_signal", code
        if not has_model_ops:
            return False, "missing_modeling_ops", code

    return True, "ok", code


def remove_file_if_exists(path: str):
    if os.path.isfile(path):
        os.remove(path)


def append_export_code(code: str, code_language: str, stl_path: str, step_path: str):
    if code_language == "pythonocc":
        raise ValueError("Implement STEP generation")
    if code_language != "cadquery":
        raise TypeError("CAD code language not supported!")

    export_block = f"""
import cadquery as cq
_candidate_objs = []
for _name in ["solid", "r", "result", "part", "body", "obj", "model"]:
    if _name in globals():
        _candidate_objs.append(globals()[_name])
if not _candidate_objs:
    raise ValueError("No CAD object found in script globals")
_cad_obj = _candidate_objs[0]
if hasattr(_cad_obj, "val"):
    _cad_obj = _cad_obj.val()
cq.exporters.export(_cad_obj, r"{stl_path}")
cq.exporters.export(_cad_obj, r"{step_path}")
"""
    return code + "\n" + export_block
    
# Writing ply file, from GenCAD/Ferdous's repo
def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)
    return

# From DeepCAD
def convert_stl_to_point_cloud(stl_path, point_cloud_path, n_points, seed=42):
    np.random.seed(seed)
    out_mesh = trimesh.load(stl_path) # load the stl as a mesh
    out_pc, _ = sample_surface(out_mesh, n_points) # convert to a point cloud
    write_ply(out_pc, point_cloud_path)
    return out_pc