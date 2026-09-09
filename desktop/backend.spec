# Run from the repository root: python -m PyInstaller desktop/backend.spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH).parent
datas = [(str(root / source), target) for source, target in [
    ("frontend/dist", "frontend/dist"), ("configs", "configs"),
    ("docs/config.md", "docs"), ("migrations", "migrations"), ("export", "export"),
    ("desktop/build-manifest.json", "."), ("desktop/THIRD_PARTY_NOTICES.txt", "."),
]]
binaries = []
hiddenimports = collect_submodules("app") + collect_submodules("uvicorn")
hiddenimports += ["openpyxl", "xlrd", "sqlalchemy.dialects.sqlite"]
for package in ("onnxruntime", "skimage"):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

a = Analysis([str(root / "desktop/backend_entry.py")], pathex=[str(root)],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    excludes=["google", "psycopg2", "tkinter", "pytest", "httpx", "IPython"],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}})
pyz = PYZ(a.pure)
# Console subsystem preserves the private stdio protocol. The native shell starts
# it with CREATE_NO_WINDOW, so users never see a terminal.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="waveatlas-backend",
    debug=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="payload")
