from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(r"F:\Exploration")
OUT = ROOT / "materials" / "finetune_pdfs"
OUT.mkdir(parents=True, exist_ok=True)

PDFS = [
    Path(r"F:\大二下专业课\机器学习\深度学习课程设计参考文献\模型微调技术探索\【NeruIPS 2022】AdaptFormer Adapting Vision Transformers for Scalable Visual Recognition.pdf"),
    Path(r"F:\大二下专业课\机器学习\深度学习课程设计参考文献\模型微调技术探索\【NeruIPS 2022】Scaling & Shifting Your Features A New Baseline for Efficient Model Tuning.pdf"),
    Path(r"F:\大二下专业课\机器学习\深度学习课程设计参考文献\模型微调技术探索\LORA LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS.pdf"),
    Path(r"F:\大二下专业课\机器学习\深度学习课程设计参考文献\模型微调技术探索\【ACL 2022】BitFit Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.pdf"),
]


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text.replace("\x00", " "))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_name(path: Path) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", path.stem).strip("_")


def main() -> None:
    index = []
    for pdf in PDFS:
        reader = PdfReader(str(pdf))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = clean(page.extract_text() or "")
            except Exception as exc:
                text = f"[EXTRACT_ERROR {type(exc).__name__}: {exc}]"
            pages.append({"page": i, "text": text})
        name = safe_name(pdf)
        txt = OUT / f"{name}.txt"
        js = OUT / f"{name}.json"
        txt.write_text(
            "\n\n".join(f"--- Page {p['page']} ---\n{p['text']}" for p in pages),
            encoding="utf-8",
        )
        js.write_text(
            json.dumps({"file": str(pdf), "pages": pages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index.append({"file": str(pdf), "pages": len(pages), "txt": str(txt)})
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
