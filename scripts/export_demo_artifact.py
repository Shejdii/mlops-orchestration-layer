import shutil
from pathlib import Path

SRC = Path("registry/liar/latest_decision.json")
DST = Path("docs/demo/latest_decision.json")

DST.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(SRC, DST)

print(f"Exported demo artifact → {DST}")
