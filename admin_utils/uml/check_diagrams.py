"""
Check that all labs have an up-to-date description.png in assets/.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from admin_utils.uml.uml_diagrams_builder import check_if_has_classes, generate_uml_diagrams

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    """
    Verify that every lab listed in project_config.json has an up-to-date
    UML diagram by comparing the committed DOT file with a freshly generated one.

    For each lab:
    - Copies the lab to a temporary directory
    - Regenerates the diagram (which produces a DOT file)
    - Compares the generated DOT with the committed one in assets/

    Exits with code 0 if all diagrams are up-to-date, 1 otherwise.
    """
    config = json.loads((ROOT / "project_config.json").read_text(encoding="utf-8"))
    all_ok = True

    for lab_info in config.get("labs", []):
        lab_name = lab_info["name"]
        lab_path = ROOT / lab_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_lab = Path(tmp_dir) / lab_name
            shutil.copytree(lab_path, tmp_lab, dirs_exist_ok=True)

            success = generate_uml_diagrams(tmp_lab)
            if not success:
                print(f"❌ Failed to regenerate diagram for {lab_name}")
                all_ok = False
                continue

            has_classes = check_if_has_classes(tmp_lab)

            if has_classes:
                project_name = lab_path.name.upper()
                committed_dot = lab_path / "assets" / f"classes_{project_name}_fixed.dot"
                generated_dot = tmp_lab / "assets" / f"classes_{project_name}_fixed.dot"
            else:
                committed_dot = lab_path / "assets" / "functions.dot"
                generated_dot = tmp_lab / "assets" / "functions.dot"

            if not committed_dot.is_file():
                print(f"❌ Missing committed DOT: {committed_dot}")
                all_ok = False
                continue

            if not generated_dot.is_file():
                print(f"❌ Generated DOT not found: {generated_dot}")
                all_ok = False
                continue

            if committed_dot.read_text() != generated_dot.read_text():
                print(f"❌ Diagram is outdated: {committed_dot}")
                all_ok = False

    if all_ok:
        print("✅ All diagrams are present and up-to-date")
        sys.exit(0)
    else:
        print("\n💡 Tip: Run the UML generator locally and commit the updated assets/*.dot")
        print("Run: python admin_utils/uml/uml_diagrams_builder.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
