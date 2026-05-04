"""
Master run script — Banking Operational Risk Intelligence Platform
Run this to execute the full pipeline end-to-end.
"""
import subprocess
import sys
import os

def run(script, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"❌ Failed: {script}")
        sys.exit(1)
    print(f"✅ Done: {label}")

if __name__ == "__main__":
    base = os.path.dirname(__file__)
    run(os.path.join(base, "src/pipeline/fdic_pipeline.py"), "Phase 1: FDIC Data Pipeline")
    run(os.path.join(base, "src/pipeline/fred_pipeline.py"), "Phase 2: FRED Interest Rate Pipeline")
    run(os.path.join(base, "src/ml/risk_engine.py"),         "Phase 3: ML Risk Engine")
    print("\n" + "="*60)
    print("  All phases complete! Launching dashboard...")
    print("="*60)
    os.system(f"streamlit run {os.path.join(base, 'src/dashboard/app.py')}")