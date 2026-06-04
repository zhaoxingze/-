from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(r"F:\Exploration")
TMP = ROOT / "outputs" / "tmp"
TMP.mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(TMP)
os.environ["TMP"] = str(TMP)
os.environ["TMPDIR"] = str(TMP)
tempfile.tempdir = str(TMP)

renderer = Path(
    r"C:\Users\zhao'xing'ze\.codex\plugins\cache\openai-primary-runtime\documents\26.601.10930\skills\documents\render_docx.py"
)
spec = importlib.util.spec_from_file_location("render_docx_skill", renderer)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

sys.argv = [
    "render_docx.py",
    str(ROOT / "模型微调技术探索_课程设计小论文.docx"),
    "--output_dir",
    str(ROOT / "outputs" / "docx_render_finetune"),
    "--emit_pdf",
    "--verbose",
]
module.main()
