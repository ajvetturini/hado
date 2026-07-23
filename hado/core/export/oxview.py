import json
from pathlib import Path

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.export.common import _resolve_output_path


def write_oxView(filename_no_extension: str, design: HadoNucleotideModel, filepath: str | Path = ".",
                 verbose: bool = False, diagnostics=None) -> None:
    """ This writes a single HadoNucleotideModel to an .oxview (which is a json) file"""
    json_data = get_oxview_json(design)
    outpath = _resolve_output_path(filepath, f"{filename_no_extension}.oxview")

    with open(outpath, "w") as f:
        json.dump(json_data, f, indent=4)

def get_oxview_json(design: HadoNucleotideModel):
    oxdna_system = design.get_oxdna_system()
    oxview_dict = oxdna_system.get_oxview_dict()
    return oxview_dict
