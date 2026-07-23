from pathlib import Path

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.export.common import _resolve_output_path
from hado.core.export.oxview import get_oxview_json


def write_oxDNA(filename_no_extension: str, design: HadoNucleotideModel,
                use_old_top: bool = False, filepath: str | Path = ".",
                verbose: bool = False, diagnostics=None) -> None:
    """
    Writes the oxDNA top and dat file using the caDNAno writer object and scadnano package to convert it to oxView.
    There is a flag to use the new (or old) topology file

    :param filepath: Filepath to write files to (does not include filename / extension)
    :type filepath: str | Path

    :param filename_no_extension: Name of the .top and .dat files that will be created
    :type filename_no_extension: str

    :param design: Resultant design model from the hado automation pipeline.
    :type design: HadoNucleotideModel

    :param use_old_top: A flag to use the old topology file format. Defaults to using the NEW oxDNA top format.
    :type use_old_top: bool

    :rtype: None
    """
    json_data = get_oxview_json(design)
    if use_old_top:
        top, dat = _old_oxdna_format(json_data)
    else:
        top, dat = _new_oxdna_format(json_data)

    topout = _resolve_output_path(filepath, f"{filename_no_extension}.top")
    datout = _resolve_output_path(filepath, f"{filename_no_extension}.dat")

    with open(topout, 'w') as f:
        f.writelines(top)

    with open(datout, 'w') as f:
        f.writelines(dat)

def get_oxDNA_strings(design: HadoNucleotideModel, use_old_top: bool = False,
                      verbose: bool = False, diagnostics=None):
    """Return oxDNA topology and configuration strings without writing files."""
    json_data = get_oxview_json(design)
    if use_old_top:
        top, dat = _old_oxdna_format(json_data)
    else:
        top, dat = _new_oxdna_format(json_data)
    return top, dat

def get_oxView_json_and_oxDNA_Strings(design: HadoNucleotideModel, use_old_top: bool = False,
                                      verbose: bool = False, diagnostics=None):
    """ The get_oxview_json has some computational cost, so this is useful if
    writing both outputs (i.e., output files from UI) """
    json_data = get_oxview_json(design)
    if use_old_top:
        top, dat = _old_oxdna_format(json_data)
    else:
        top, dat = _new_oxdna_format(json_data)
    return json_data, (top, dat)

def _old_oxdna_format(oxview_json_data: dict):
    """ Converts the oxView json dictionary (which is a list of strands from scadnano)
     to the dat / top text for the OLD oxDNA format """
    strands = oxview_json_data['systems'][0]['strands']
    num_strands = len(strands)
    num_nts = 0
    strand_texts = []
    bbox = oxview_json_data['box']
    dat_text = ['t = 0\n', f'b = {bbox[0]} {bbox[1]} {bbox[2]}\n', 'E = 0 0 0\n']
    for s in strands:
        monomers = s['monomers']
        num_nts += len(monomers)
        for m in monomers:
            dat_text.append(f"{m['p'][0]} {m['p'][1]} {m['p'][2]} "
                            f"{m['a1'][0]} {m['a1'][1]} {m['a1'][2]} "
                            f"{m['a3'][0]} {m['a3'][1]} {m['a3'][2]} 0 0 0 0 0 0\n")

            # Legacy topology format stores explicit 3' and 5' neighbours in that order.
            full_strand_text = f"{s['id']} {m['type']} {m.get('n3', -1)} {m.get('n5', -1)}\n"
            strand_texts.append(full_strand_text)

    first_line = f'{num_nts} {num_strands}\n'
    strand_texts.insert(0, first_line)
    return strand_texts, dat_text

def _new_oxdna_format(oxview_json_data: dict):
    """ Converts the oxView json dictionary (which is a list of strands from scadnano)
     to the dat / top text for the NEW oxDNA format """
    strands = oxview_json_data['systems'][0]['strands']
    num_strands = len(strands)
    num_nts = 0
    strand_texts = []
    bbox = oxview_json_data['box']
    dat_text = ['t = 0\n', f'b = {bbox[0]} {bbox[1]} {bbox[2]}\n', 'E = 0 0 0\n']
    for s in strands:
        monomers = s['monomers']
        num_nts += len(monomers)
        strand_text = ''
        for m in monomers:
            dat_text.append(f"{m['p'][0]} {m['p'][1]} {m['p'][2]} "
                            f"{m['a1'][0]} {m['a1'][1]} {m['a1'][2]} "
                            f"{m['a3'][0]} {m['a3'][1]} {m['a3'][2]} 0 0 0 0 0 0\n")
            strand_text += m['type']

        full_strand_text = f"{strand_text} id={s['id']} type=DNA circular=false\n"
        strand_texts.append(full_strand_text)

    first_line = f'{num_nts} {num_strands} 5->3\n'
    strand_texts.insert(0, first_line)
    return strand_texts, dat_text
