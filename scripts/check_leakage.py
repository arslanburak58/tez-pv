"""
Leakage Kontrol Scripti
=======================
Scaler, imputer ve encoder .fit() çağrılarını tara.
Bunların SADECE train setinde kullanıldığından emin ol.

Çalıştır: python scripts/check_leakage.py
       ya da: make leakage
"""
import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
RISK_METHODS = {"fit", "fit_transform"}
SKIP_DIRS = {"tez-env", "venv", ".git", "__pycache__", "scripts"}
SKIP_FILES = {"check_leakage.py"}

issues = []
for py_file in ROOT.rglob("*.py"):
    # Atlanacak dizinler
    if any(skip in py_file.parts for skip in SKIP_DIRS):
        continue
    if py_file.name in SKIP_FILES:
        continue
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            method = getattr(node.func, "attr", None)
            if method in RISK_METHODS:
                line = source.splitlines()[node.lineno - 1].strip()
                issues.append(
                    f"  ⚠️  {py_file.relative_to(ROOT)}:{node.lineno}\n"
                    f"       → {line}"
                )

print("\n" + "=" * 50)
print("  Leakage Kontrol Raporu")
print("=" * 50)
if issues:
    print(f"\n  {len(issues)} .fit() çağrısı bulundu — kontrol et:\n")
    print("\n".join(issues))
    print("\n  Not: Bu çağrıların SADECE train setinde olduğundan")
    print("  emin ol. val/test için sadece .transform() kullan.")
else:
    print("\n  ✓ Belirgin leakage riski tespit edilmedi.")
print("=" * 50 + "\n")
