"""добавляет в конец 04_task1_rating_mlp.ipynb markdown-ячейку с указателем на финальный бленд,
чтобы ноутбук не противоречил тому, что итоговая модель проекта собрана в 07_task1_blend.ipynb.
правка идемпотентна: повторный запуск ничего не дублирует.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "04_task1_rating_mlp.ipynb"
MARK = "финальная модель проекта это гетерогенный бленд"
NOTE = [
    "## финальная модель проекта\n",
    "\n",
    f"{MARK} этой сети с порядковой головой CORAL, GBDT, recency-вариантом и сетью на обогащённых ",
    "prior-признаках, веса членов подобраны на val. ансамбль выше это сильнейший одиночный член бленда ",
    "(test RMSE ~1.045), а бленд снижает test RMSE до 1.0424. сборка и разбор бленда в ",
    "`scripts/task1_finalize.py` и `notebooks/07_task1_blend.ipynb`.",
]


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("cell_type") == "markdown" and any(MARK in s for s in c.get("source", [])):
            print("указатель уже есть, пропускаем")
            return
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": NOTE})
    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("добавлена markdown-ячейка с указателем на бленд")


if __name__ == "__main__":
    main()
