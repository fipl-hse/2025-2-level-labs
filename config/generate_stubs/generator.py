"""
Generator of stubs for existing lab implementation.
"""

import ast
from _ast import alias, stmt
from pathlib import Path

import ast_comments
from tap import Tap

from config.console_logging import get_child_logger

logger = get_child_logger(__file__)


class NoDocStringForAMethodError(Exception):
    """
    Error for a method that lacks docstring.
    """


def remove_implementation_from_function(
    original_declaration: ast.stmt, parent: ast.ClassDef | None = None
) -> None:
    """
    Remove reference implementation.

    Args:
        original_declaration (ast.stmt): Original declaration
        parent (ast.ClassDef | None): Parent
    """
    if not isinstance(original_declaration, ast.FunctionDef):
        return

    expr = original_declaration.body[0]
    if not isinstance(expr, ast.Expr) and (
        not hasattr(expr, "value") or not isinstance(getattr(expr, "value"), ast.Constant)
    ):
        raise NoDocStringForAMethodError(
            f"You have to provide docstring for a method "
            f'{parent.name + "." if parent is not None else ""}'
            f"{original_declaration.name}"
        )

    opening_files = []
    for decl in original_declaration.body:
        if isinstance(decl, ast.Expr) and "# stubs: keep" in ast.unparse(decl.value):
            opening_files.extend(original_declaration.body[1:])

        if isinstance(decl, ast.With) and decl not in opening_files:
            if not ast.unparse(decl.items[0].context_expr.args):
                continue
            if "assets" in ast.unparse(decl.items[0].context_expr.args[0]):  # type: ignore
                opening_files.append(decl)

        if isinstance(decl, ast.Assert):
            add_none = ast.parse("result = None")
            opening_files.extend([add_none, decl])  # type: ignore
    original_declaration.body[1:] = opening_files


# pylint: disable=too-many-branches,too-many-statements,too-many-locals
def cleanup_code(source_code_path: Path) -> str:
    """
    Remove implementation based on AST parsing of code.

    Args:
        source_code_path (Path): Path to source code

    Returns:
        str: Implementation without AST parsing of code
    """
    with source_code_path.open(encoding="utf-8") as file:
        data = ast.parse(file.read(), source_code_path.name, type_comments=True)

    with source_code_path.open(encoding="utf-8") as file:
        data_2 = ast_comments.parse(file.read(), source_code_path.name)

    accepted_modules: dict[str, list[str]] = {"typing": ["*"], "pathlib": ["Path"]}

    if source_code_path.name == "pipeline.py":
        accepted_modules["networkx"] = ["DiGraph"]
        accepted_modules["core_utils.pipeline"] = [
            "PipelineProtocol",
            "LibraryWrapper",
            "AbstractCoNLLUAnalyzer",
            "UDPipeDocument",
            "StanzaDocument",
            "CoNLLUDocument",
            "TreeNode",
            "UnifiedCoNLLUDocument",
        ]
        accepted_modules["core_utils.article.article"] = ["Article"]
    elif (
        "lab_4_retrieval_w_clustering" in str(source_code_path)
        and source_code_path.name == "main.py"
    ):
        accepted_modules["lab_3_ann_retriever.main"] = [
            "BasicSearchEngine",
            "Tokenizer",
            "Vector",
            "Vectorizer",
        ]

    new_decl: list[stmt] = []

    for decl_index, decl_2 in enumerate(data_2.body):
        if isinstance(decl_2, ast_comments.Comment):
            data.body.insert(decl_index, decl_2)

        if isinstance(decl_2, ast.ClassDef):
            for class_index, class_decl in enumerate(decl_2.body):
                if isinstance(class_decl, ast_comments.Comment) and "#: " in class_decl.value:
                    data.body[decl_index].body.insert(class_index, class_decl)

    for decl in data.body:
        if (
            isinstance(decl, ast.AsyncFunctionDef)
            or isinstance(decl, ast.ClassDef)
            and decl.name == "Query"
        ):
            decl = []  # type: ignore

        if source_code_path.name == "service.py" and isinstance(decl, ast.Assign):
            if source_code_path.parent.name == "lab_7_llm":
                decl = ast.parse("app, pipeline = None, None")  # type: ignore
            elif source_code_path.parent.name == "lab_8_sft":
                decl = ast.parse(  # type: ignore
                    "app, pre_trained_pipeline, fine_tuned_pipeline = None, None, None"
                )

        if isinstance(decl, (ast.Import, ast.ImportFrom)):
            if (module_name := getattr(decl, "module", None)) is None:
                module_name = decl.names[0].name

            if module_name not in accepted_modules:
                continue

            if isinstance(decl, ast.ImportFrom):
                accepted_names = accepted_modules.get(module_name, [])
                names_to_import = [
                    name
                    for name in decl.names
                    if name.name in accepted_names or "*" in accepted_names
                ]

                if not names_to_import:
                    continue

                new_decl.append(
                    ast.ImportFrom(
                        module=module_name,
                        names=[alias(name=name.name) for name in names_to_import],
                    )
                )
                continue

        if isinstance(decl, ast.ClassDef) and isinstance(ast.get_docstring(decl), str):
            if "Note: remove" in ast.get_docstring(decl):  # type: ignore
                decl = []  # type: ignore
            else:
                for class_decl in decl.body:
                    if not isinstance(class_decl, ast.FunctionDef):
                        continue

                    docstring = ast.get_docstring(class_decl)
                    if docstring is None:
                        raise ValueError(
                            f"{source_code_path.parent.name}.{source_code_path.stem}."
                            f"{decl.name}.{class_decl.name} does not have a docstring!"
                        )

                    if "Note: remove" in ast.get_docstring(class_decl):  # type: ignore
                        decl.body[decl.body.index(class_decl)] = []  # type: ignore

        if isinstance(decl, ast.ClassDef) and decl.bases:
            name = decl.bases[0]
            if (
                decl.bases
                and isinstance(name, ast.Name)
                and hasattr(name, "id")
                and getattr(name, "id") == "Exception"
            ):
                decl = []  # type: ignore

        if isinstance(decl, ast.ClassDef):
            for class_decl in decl.body:
                remove_implementation_from_function(class_decl, parent=decl)

        remove_implementation_from_function(decl)

        new_decl.append(decl)

    data.body = list(new_decl)
    return ast_comments.unparse(data)  # type: ignore


class ArgumentParser(Tap):
    """
    Types for CLI interface of a module.
    """

    source_code_path: str
    target_code_path: str


def main() -> None:
    """
    Entrypoint for stub generation.
    """
    args = ArgumentParser().parse_args()

    res_stub_path = Path(args.target_code_path)
    res_stub_path.parent.mkdir(parents=True, exist_ok=True)

    source_code = cleanup_code(Path(args.source_code_path))

    with res_stub_path.open(mode="w", encoding="utf-8") as file:
        logger.info(f"Writing to {res_stub_path}")
        file.write(source_code)


if __name__ == "__main__":
    main()
