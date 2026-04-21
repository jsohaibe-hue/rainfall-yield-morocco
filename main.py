"""
main.py
Point d'entrée principal du projet.
Lance toutes les étapes dans l'ordre.
"""

import os
import sys
import time

def run_step(script, label):
    print(f"\n{'─'*60}")
    print(f"🚀 {label}")
    print('─'*60)
    start = time.time()
    ret = os.system(f"python3 scripts/{script}")
    elapsed = time.time() - start
    if ret == 0:
        print(f"   ✅ Terminé en {elapsed:.1f}s")
    else:
        print(f"   ❌ Erreur dans {script}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("  ANALYSE SPATIO-TEMPORELLE — PLUVIOMÉTRIE MAROC")
    print("  IAV Hassan II × UCLouvain — LBRTI2101")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    run_step("generate_data.py",   "1/3 — Génération des données")
    run_step("temporal_analysis.py", "2/3 — Analyse temporelle")
    run_step("spatial_analysis.py",  "2/3 — Analyse spatiale")
    run_step("ml_prediction.py",     "3/3 — Machine Learning")

    print("\n" + "="*60)
    print("  ✅ PROJET COMPLET — tous les fichiers dans outputs/")
    print("="*60)
    outputs = sorted(os.listdir("outputs"))
    for f in outputs:
        size = os.path.getsize(f"outputs/{f}")
        print(f"    {f:45s}  {size/1024:.0f} Ko")
