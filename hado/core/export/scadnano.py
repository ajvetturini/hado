from pathlib import Path

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel


def write_scadnano(filename_no_extension: str, design: HadoNucleotideModel, filepath: str | Path = ".") -> None:
    """
    Writes the scadnano SC file using the caDNAno writer object.

    :param filepath: Filepath to write files to (does not include filename / extension)
    :type filepath: str | Path

    :param filename_no_extension: Name of the .sc file that will be created
    :type filename_no_extension: str

    :param design: Resultant design model from the hado automation pipeline.
    :type design: HadoNucleotideModel

    :rtype: None
    """
    fname = filename_no_extension + '.sc'
    sc_design = design.get_sc_design()
    if sc_design is None:
        raise Exception('ERROR: scadnano design not set. Use HadoNucleotideModel.to_scadnano() first.')

    sc_design.write_scadnano_file(filename=fname, directory=filepath)

def get_scadnano_json(design: HadoNucleotideModel):
    """Return scadnano JSON for a model that already has scadnano data set."""
    sc_design = design.get_sc_design()
    if sc_design is None:
        raise Exception('ERROR: scadnano design not set. Use HadoNucleotideModel.to_scadnano() first.')
    return sc_design.to_json()
