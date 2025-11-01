"""
UML Diagram Generator for Labs

Automatically generates structural diagrams for Python labs using hybrid approach:
- Class diagrams via pyreverse for OOP labs
- Module dependency diagrams via AST analysis for procedural labs

Workflow:
1. Discovers functional Python files (excludes tests/__pycache__)
2. Detects if lab contains classes
3. Generates appropriate diagram type:
   - Class diagrams (pyreverse + Graphviz) for OOP labs
   - Module dependency graphs (AST parsing + Graphviz) for procedural labs
4. Outputs standardized description.png in each lab's assets/ folder

Tools Integration:
- pyreverse: deep class structure analysis
- AST parsing: module dependency mapping
- Graphviz: universal diagram rendering
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

PROJECT_CONFIG_PATH = Path(__file__).parent.parent.parent / "project_config.json"


def get_functional_files(lab_folder: Path) -> List[Path]:
    """
    Get all .py files in lab folder, excluding tests and __pycache__.

    Args:
        lab_folder (Path): Path to the lab folder.

    Returns:
        List[Path]: List of paths to functional Python files (excluding tests and cache).
    """
    functional_files = []
    for py_file in lab_folder.rglob("*.py"):
        if (
            any(part.startswith("test") or part == "tests" for part in py_file.parts)
            or "__pycache__" in py_file.parts
            or py_file.name.startswith("test_")
            or py_file.name.endswith("_test.py")
        ):
            continue
        functional_files.append(py_file)
    return functional_files


def is_module_in_lab(module_name: str, lab_folder: Path) -> bool:
    """
    Check if a module exists in the lab folder.

    Used during import analysis to determine whether an imported name
    corresponds to a local module (i.e. part of the lab) rather than
    a built-in or third-party package.

    Args:
        module_name (str): Name of the module to be checked.
        lab_folder (Path): Path to the lab folder.

    Returns:
        bool: True if the module exists as a .py file or package in the lab folder,
        False otherwise.
    """
    if not module_name:
        return False

    possible_paths = [lab_folder / f"{module_name}.py", lab_folder / module_name / "__init__.py"]

    return any(path.exists() for path in possible_paths)


def check_if_has_classes(lab_folder: Path) -> bool:
    """
    Check if any functional .py file contains a class definition.

    Args:
        lab_folder (Path): Path to the lab folder.

    Returns:
        bool: True if there are classes, otherwise False.
    """
    for py_file in get_functional_files(lab_folder):
        try:
            content = py_file.read_text(encoding="utf-8")
            if "class " in content:
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


def _run_pyreverse(
    project_name: str, output_dir: Path, file_list: list[str], cwd: Path
) -> tuple[str, str, int]:
    """
    Run pyreverse to generate DOT class diagram files.

    Args:
        project_name (str): Name of the project used for DOT file naming.
        output_dir (Path): Directory where generated DOT files will be saved.
        file_list (list[str]): List of relative paths to Python source files to analyze.
        cwd (Path): Working directory for the pyreverse command execution.

    Returns:
        tuple[str, str, int]: stdout, stderr, and exit code of the pyreverse process.
    """
    from config.cli_unifier import _run_console_tool  # pylint: disable=import-outside-toplevel

    return _run_console_tool(
        "pyreverse",
        ["-o", "dot", "-p", project_name, "-d", str(output_dir), "-A", *file_list],
        cwd=cwd,
    )


def _run_dot(input_path: Path, output_path: Path) -> tuple[str, str, int]:
    """
    Render a DOT file to PNG using Graphviz dot command.

    Converts the given DOT diagram file into a PNG image using the dot utility.

    Args:
        input_path (Path): Path to the input DOT file.
        output_path (Path): Path where the resulting PNG image will be saved.

    Returns:
        tuple[str, str, int]: stdout, stderr, and exit code of the dot process.
    """
    from config.cli_unifier import _run_console_tool  # pylint: disable=import-outside-toplevel

    return _run_console_tool(
        "dot", ["-Tpng", "-Gid=uml_diagram", str(input_path), "-o", str(output_path)]
    )


def generate_class_diagram(lab_folder: Path, output_dir: Path) -> bool:
    """
    Generate UML class diagram using pyreverse for functional files only.

    1. Collects all non-test Python files in the lab.
    2. Runs pyreverse on them to produce a DOT class diagram.
    3. Post-processes the DOT file to improve formatting.
    4. Converts it to PNG and saves as 'description.png'.
    5. Keeps only the fixed DOT file for reproducibility; removes raw pyreverse output.

    Args:
        lab_folder (Path): Path to the lab directory containing Python source files.
        output_dir (Path): Directory where the diagram (description.png) is saved.

    Returns:
        bool: True if the diagram was successfully generated and saved, otherwise False.
    """
    project_name = lab_folder.name.upper()
    functional_files = sorted(get_functional_files(lab_folder))

    if not functional_files:
        print(f"  No functional Python files found in {lab_folder.name}")
        return False

    # Prepare list of files for pyreverse
    file_list = [str(f.relative_to(lab_folder)) for f in functional_files]
    file_list.sort()
    _, stderr, returncode = _run_pyreverse(project_name, output_dir, file_list, lab_folder)

    if returncode != 0:
        print(f"  pyreverse failed with return code {returncode}")
        if stderr:
            print(f"  Error: {stderr.strip()}")
        # Clean up any partial output
        (output_dir / f"classes_{project_name}.dot").unlink(missing_ok=True)
        (output_dir / f"packages_{project_name}.dot").unlink(missing_ok=True)
        return False

    original_file = output_dir / f"classes_{project_name}.dot"
    packages_file = output_dir / f"packages_{project_name}.dot"
    if not original_file.exists():
        print(f"  Generated DOT file not found: {original_file}")
        packages_file.unlink(missing_ok=True)
        return False

    fixed_file = output_dir / f"classes_{project_name}_fixed.dot"
    process_and_fix_dot(original_file, fixed_file)

    png_path = output_dir / "description.png"
    _, stderr, returncode = _run_dot(fixed_file, png_path)

    original_file.unlink(missing_ok=True)
    packages_file.unlink(missing_ok=True)

    if returncode != 0:
        stderr_msg = stderr.strip() if stderr else "no stderr"
        print(f"  dot command failed: {stderr_msg}")
        return False

    return png_path.exists()


def process_and_fix_dot(original_path: Path, fixed_path: Path) -> None:
    """
    Fix dot file to improve class formatting.

    Args:
        original_path (Path): Path to the original DOT file produced by pyreverse.
        fixed_path (Path): Path where the cleaned/fixed DOT file will be saved.
    """
    with open(original_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    result_lines = []
    for line in lines:
        if 'label="{' in line and "|" in line:
            result_lines.append(fix_class_line(line))
        else:
            result_lines.append(line)

    with open(fixed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines) + "\n")


def fix_class_line(line: str) -> str:
    """
    Clean and deduplicate fields/methods in a class dot line.

    Args:
        line (str): A raw DOT line containing a class label.

    Returns:
        str: The cleaned and reformatted DOT line with deduplicated fields/methods.
    """
    start = line.find('label="{') + 8
    end = line.find('}"', start)
    if start <= 8 or end <= start:
        return line

    parts = line[start:end].split("|", 2)
    if len(parts) < 3:
        return line

    name, fields_sec, methods_sec = parts

    fields = [f.strip() for f in fields_sec.split("\\l") if f.strip()]
    seen = set()
    unique_fields = []
    for f in fields:
        fname = f.split(":")[0].strip()
        if fname and fname not in seen:
            seen.add(fname)
            unique_fields.append(f"\u200b{f}")

    methods = [m.strip() for m in methods_sec.split("\\l") if m.strip()]
    fixed_methods = [f"\u200b{m}" for m in methods]

    new_content = name + "|" + "\\l".join(unique_fields) + "|" + "\\l".join(fixed_methods)
    return line[:start] + new_content + line[end:]


def generate_module_diagram(lab_folder: Path, output_dir: Path) -> bool:
    """
    Generate function-level diagram for labs without classes.
    Shows functions defined in main.py and their relationships.

    Args:
        lab_folder (Path): Path to the lab directory.
        output_dir (Path): Directory where the diagram (description.png) is be saved.

    Returns:
        bool: True if the diagram was successfully generated and saved, otherwise False.
    """
    main_file = lab_folder / "main.py"
    if not main_file.exists():
        print(f"  main.py not found in {lab_folder.name}")
        return False

    functions = extract_functions(main_file)
    if not functions:
        return False

    dot_header = [
        "digraph FunctionDependencies {",
        '  rankdir="LR";',
        '  node [shape=box, style=filled, fillcolor="#E0F0FF"];',
        '  main [label="main.py", shape=folder, fillcolor="#FFE0E0"];',
    ]

    function_nodes = []
    for func_name in functions:
        function_nodes.append(f'  "{func_name}" [label="{func_name}()"];')
        function_nodes.append(f'  main -> "{func_name}";')

    dot_content = dot_header + function_nodes + ["}"]

    dot_path = output_dir / "functions.dot"
    png_path = output_dir / "description.png"

    try:
        dot_path.write_text("\n".join(dot_content) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"  Error writing DOT file: {e}")
        return False

    try:
        result = subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)], capture_output=True, check=False
        )
    except subprocess.SubprocessError as e:
        print(f"  Error running dot command: {e}")
        return False

    if result.returncode != 0:
        stderr_msg = result.stderr.strip() if result.stderr else "no stderr"
        print(f"  dot command failed: {stderr_msg}")
        return False

    return png_path.exists()


def extract_functions(py_file: Path) -> List[str]:
    """
    Parses the given Python file and collects names of all top-level
    function definitions (excluding nested or lambda functions).

    Args:
        py_file (Path): Path to the Python source file.

    Returns:
        List[str]: Sorted list of function names defined in the file.
    """
    functions = []
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
    except (SyntaxError, UnicodeDecodeError):
        pass
    return sorted(functions)


def generate_uml_diagrams(lab_folder: Path) -> bool:
    """
    Generate appropriate UML diagram for a lab.
    Automatically selects diagram type based on whether the lab contains classes.

    Args:
        lab_folder (Path): Path to the lab directory.

    Returns:
        bool: True if diagram generation succeeded, otherwise False.
    """
    lab_folder_resolved = Path(lab_folder).resolve()
    assets_dir = lab_folder_resolved / "assets"
    output_dir = assets_dir if assets_dir.is_dir() else lab_folder_resolved
    output_dir.mkdir(exist_ok=True)

    has_classes = check_if_has_classes(lab_folder_resolved)
    diagram_type = "class diagram" if has_classes else "function-level diagram"
    print(f"  Generating {diagram_type}...")

    if has_classes:
        success = generate_class_diagram(lab_folder_resolved, output_dir)
    else:
        success = generate_module_diagram(lab_folder_resolved, output_dir)

    if success:
        print("  ✅ Diagram generated successfully")
    else:
        print("  ❌ Failed to generate diagram")

    return success


def main() -> None:
    """
    Generate diagrams for all labs in project_config.json.

    Reads the project configuration, iterates through all registered labs,
    and triggers diagram generation for each one. Skips missing lab folders.
    """
    if not PROJECT_CONFIG_PATH.exists():
        print(f"Config file not found: {PROJECT_CONFIG_PATH}")
        return

    with open(PROJECT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    root_dir = PROJECT_CONFIG_PATH.parent
    labs = config.get("labs", [])

    print(f"Found {len(labs)} labs in config")

    for lab_info in labs:
        lab_name = lab_info["name"]
        lab_path = root_dir / lab_name
        if not lab_path.exists():
            print(f"❌ Lab folder not found: {lab_path}")
            continue

        print(f"Processing {lab_name}...")
        if not generate_uml_diagrams(lab_path):
            print(f"Failed to generate diagram for {lab_name}")


if __name__ == "__main__":
    main()
