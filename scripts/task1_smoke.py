"""быстрый smoke-тест пайплайна: prepare + один короткий прогон, ловит ошибки до полного запуска."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import task1_lib as L

L.MAX_EPOCHS = 2
L.PATIENCE = 2
D = L.prepare()
print("device", L.device, "строк", len(D["df"]), "cat_vocab", D["n_cat_vocab"], "biz_vocab", D["n_biz_vocab"])
Inter = D["models"]["InteractionMLP"]
f = lambda du, db, dc: Inter(du, db, dc, d_cat=32, d_branch=48, hidden=128, top=(256, 128), out=5)
r = L.run_experiment("smoke cls big", f, "CE", L.U_FULL, L.B_FULL, L.C_FULL,
                     D["splits"], D["y_val"], D["n_cat_vocab"], seed=42)
print("val_pred", r["val_pred"].shape, "test_pred", r["test_pred"].shape,
      "test RMSE", round(L.rmse(D["y_te"], r["test_pred"]), 4))
print("smoke ok")
