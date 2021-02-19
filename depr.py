import sys
import os
from pathlib import Path

# parameter handling
path = 0
if len(sys.argv)>1:
    path = sys.argv[1]
else:
    raise RuntimeError("missing argument")

src = Path(path)
if not src.exists():
    raise RuntimeError("path does not exist")

if src.parts[0] != "pycqed":
    raise RuntimeError("path should start with 'pycqed'")


dst = Path('deprecated') / src.parent
print(f"mkdir {str(dst)}")
dst.mkdir(parents=True, exist_ok=True)

cmd = f"git mv {str(src)} {str(dst)}"
print(cmd)
os.system(cmd)
