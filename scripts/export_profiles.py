"""Export the exact profile table used by the experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT
from sim.edca.profile_table import export_profiles, load_profiles

if __name__ == "__main__":
    table = export_profiles(load_profiles(ROOT / "configs/profiles.json"), ROOT / "results/summary")
    target = ROOT / "reports/EDCA_Profile参数表.md"
    target.write_text("# EDCA Profile 完整参数表\n\n" + table + "\n", encoding="utf-8")
    design=ROOT/"reports/实验设计说明.md"
    if design.exists():
        prefix=design.read_text(encoding="utf-8").split("<!-- GENERATED_PROFILE_TABLE -->")[0].rstrip()
        design.write_text(prefix+"\n\n<!-- GENERATED_PROFILE_TABLE -->\n\n"+table+"\n",encoding="utf-8")
    print(target)
