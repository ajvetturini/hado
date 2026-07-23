import json
from pathlib import Path
import numpy as np

from hado.core.utils import Geometry, ScaffoldArgs, StapleArgs, MIN_ALLOWABLE_EDGE_LENGTH_NTS
from hado.core.automation.model.nucleotide_model import HadoNucleotideModel, initialize_base_design
from hado.core.automation.routing.scaffold_routing import perform_scaffold_routing
from hado.core.automation.connections.connect_bundles import optimize_connections
from hado.core.automation.mitering.mitering import miter_design
from hado.core.automation.autostaple import autostaple_hollowframe
from hado.core.automation.mitering.scaffold_overhangs import add_scaffold_3p_overhangs
from hado.core.automation.pipeline.sequence import sequence_design
from hado.core.automation.routing.lattice import get_lattice_config
from hado.core.automation.pipeline.config import PipelineConfig
from hado.core.automation.pipeline.types import (
    AutostapleResult,
    BaseDesignResult,
    ConnectionOptimizationResult,
    MiteringResult,
    PipelineDiagnostics,
    emit_runtime_message,
    ScaffoldRoutingResult,
)
from hado.core.export import (write_cadnano, write_scadnano, write_oxDNA, write_oxView,
                              get_cadnano_json, get_scadnano_json, get_oxview_json, get_oxDNA_strings,
                              get_oxView_json_and_oxDNA_Strings)

class HadoManager:
    """
    This is the main data object that manages (runs) the Hollowframe DNA origami (hado) design automation pipeline.
    Users must specify the geometry (alongside a design name) to run the pipeline, and they can optionally specify
    different scaffold / staple args (see ScaffoldArgs and StapleArgs dataclasses, allowing you to do things such as
    specify a custom scaffold sequence).

    This object stores the nucleotide-level model (HadoNucleotideModel) as an attribute, which can be exported
    to various CAD formats (e.g., cadnano, scadnano, oxDNA / oxView) or have the staple sequences written to CSV.
    The manager also has a save/load functionality that allows you to save the state of the manager (including the
    nucleotide model) to a JSON file. This allows you to interoperate with the web-based GUI as a way to manually
    design the input geometries and then load them into a Pythonic environment for further analysis / refinement /
    export.
    """
    def __init__(self,
                 design_name: str,
                 geometry: Geometry,
                 scaffold_args: ScaffoldArgs = ScaffoldArgs(),
                 staple_args: StapleArgs = StapleArgs(),
                 verbose: bool = False
                 ):
        self.design_name = design_name
        self.geometry = geometry
        self.scaffold_args = scaffold_args or ScaffoldArgs(scaffold_sequence='m13', lattice_type='dna_honeycomb')
        self.staple_args = staple_args or StapleArgs()
        self.verbose = verbose
        self.nucleotide_level_model: HadoNucleotideModel | None = None

    def run(self, **kwargs) -> PipelineDiagnostics:
        """
        Main entry point running the hado design automation algorithm. Overall, there are a variety of kwargs (examples
        below) that can be used for diagnostics purposes or to skip certain portions of the algorithm.

        Overall, this method is built such that you can sub-class this manager
        (e.g., class CustomDesignMotif(HadoManager)), and re-define any (or all) of scaffold_routing_algorithm,
        #  initialize_hado_design, initialize_hado_design, vertex_matching_algorithm, mitering_algorithm,
        #  or autostaple_algorithm. As long as those methods return the proper *Result (e.g., ScaffoldRoutingResult)
        #  objects, then you can replace the components as you want.

        :param kwargs: Additional arguments that can skip certain steps of algorith, some example are:
            - skip_stapling: bool (skip the stapling procedure. Staple nucleotides are placed wherever scaffold is)
            - skip_mitering: bool (skip the mitering algorithm application to helix bundles)
            - prepare_exports: bool (prepare scadnano/oxDNA export state after running; defaults to True)
            - override_staple_autobreak_limit: bool (if design is very large # of nts in scaffold (50K+) then the
                GUI is not able to be used. This flag can be set True to allow for the EXPENSIVE search for optimal
                staple breaks to take place. Note that this can take VERY long (30's of minutes or more depending
                on how large you make it from my testing)
            - max_cross_section_size_override: bool (if design has a very large cross section (>48 helices) then the
                GUI is not able ot be used. This flag can be set True to allow for an EXPENSIVE search for optimal
                cross-section to be designed, but the algorithm is rather inefficient (because DNA origami would very
                rarely be larger than this cross-section size due to scaffold limitations).

        :return: The diagnostics of the run (note, the nucleotide-level model is stored internally to the Manager).
        """
        pipeline_config = PipelineConfig.from_kwargs(kwargs)
        stage_kwargs = pipeline_config.to_kwargs()

        diagnostics = PipelineDiagnostics()
        stage_kwargs["diagnostics"] = diagnostics

        routing = self.scaffold_routing_algorithm()
        diagnostics.record("routing", num_edges=len(routing.edge_xsect_definitions),
                           scaffold_routing_result=routing)

        base_design = self.initialize_hado_design(routing.edge_xsect_definitions, **stage_kwargs)

        design = base_design.design
        diagnostics.record("base_design", scaffold_shape=design.get_scaffold_nucleotides().shape,
                           base_design_result=base_design)

        connections = self.vertex_matching_algorithm(design, **stage_kwargs)
        design.set_connected_helices(connections.optimal_connections)
        diagnostics.record("connections", num_vertices=len(connections.optimal_connections),
                           num_rotations=len(connections.best_state), connection_optimization_result=connections)

        if pipeline_config.control.skip_mitering:
            emit_runtime_message(
                "Skipping mitering step as specified by user...",
                diagnostics=diagnostics,
                verbose=self.verbose,
            )
            mitered = MiteringResult(final_positions={}, final_nts=None)
            diagnostics.record("mitering", skipped=True)
        else:
            mitered = self.mitering_algorithm(design, connections, **stage_kwargs)
            design.set_scaffold_nucleotides(mitered.final_nts)
            design.set_staple_nucleotides(mitered.final_nts)
            diagnostics.record("mitering", skipped=False, scaffold_shape=mitered.final_nts.shape,
                               mitering_result=mitered)

        design.populate_scaffold_crossovers(connections.scaffold_helix_connections, **stage_kwargs)
        diagnostics.record("scaffold_crossovers", num_crossovers=len(design.get_scaffold_crossovers()))

        scaffold_overhang_nts = 0
        if not stage_kwargs.get("skip_scaffold_overhangs", False):
            scaffold_overhang_nts = add_scaffold_3p_overhangs(
                design,
                mitered.final_positions,
                self.staple_args,
                self.verbose,
                diagnostics,
            )
        diagnostics.record("scaffold_overhangs", num_nucleotides=scaffold_overhang_nts)
        scaffold_overhang_mask = (
            design.get_scaffold_nucleotides().astype(bool)
            & ~design.get_staple_nucleotides().astype(bool)
        )

        if pipeline_config.control.skip_stapling:
            emit_runtime_message(
                "Skipping stapling step as specified by user...",
                diagnostics=diagnostics,
                verbose=self.verbose,
            )
            diagnostics.record("autostaple", skipped=True)
        else:
            autostapled = self.autostaple_algorithm(design, connections, mitered, **stage_kwargs)
            design = autostapled.design
            design.set_rotated_helix_positions(mitered.final_positions)
            diagnostics.record("autostaple", skipped=False, autobreak_cost=autostapled.autobreak_cost,
                               autostaple_result=autostapled)

        self._verify_scaffold_overhangs_unpaired(design, scaffold_overhang_mask)
        self.set_nucleotide_model(design, prepare_exports=pipeline_config.control.prepare_exports)

        return diagnostics


    def scaffold_routing_algorithm(self):
        """ This method validates how many helices are assigned to each edge of the input geometry and which
        direction the helices are running
        """
        return ScaffoldRoutingResult(perform_scaffold_routing(self.geometry, self.scaffold_args))

    def initialize_hado_design(self, edge_xsect_definitions, **stage_kwargs):
        """ Based on the verified scaffold routing algorithm, this method will create a HadoNucleotideModel
        object that will be used with mitering and stapling
        """
        return BaseDesignResult(initialize_base_design(self.geometry, self.scaffold_args, self.staple_args,
                                                       edge_xsect_definitions, **stage_kwargs))

    def vertex_matching_algorithm(self, design, **stage_kwargs):
        """ Determines how the 3' and 5' ends of each helix bundle are connected. By default, this runs the rotation +
        Hungarian algorithm for matching the ends prior to mitering.
        """
        return ConnectionOptimizationResult(*optimize_connections(design, **stage_kwargs))

    def mitering_algorithm(self, design, connections, **stage_kwargs):
        """ Miters the ends of the helix bundles to ensure conformal fit between helix bundles
         Note that AFTER this function is executed there will be further ssDNA scaffold nucleotides added that act to
         relieve backbone torsion. This is performed in the main run() method and should NOT be handled by any
         custom mitering algorithm used here.
         """
        return self._run_mitering_with_half_turn_retry(design, connections.best_state,
                                                       connections.optimal_connections, **stage_kwargs)

    def autostaple_algorithm(self, design, connections, mitered, **stage_kwargs):
        """ Handles the stapling portion of the design automation and prescribes where staple crossovers / breakpoints
        are located.
        """
        return AutostapleResult(*autostaple_hollowframe(design, connections.optimal_connections,
                                                        mitered.final_positions, **stage_kwargs))

    @staticmethod
    def _verify_scaffold_overhangs_unpaired(design: HadoNucleotideModel, scaffold_overhang_mask: np.ndarray):
        if scaffold_overhang_mask.size == 0 or not np.any(scaffold_overhang_mask):
            return
        staple_nucleotides = design.get_staple_nucleotides().astype(bool)
        if np.any(staple_nucleotides[scaffold_overhang_mask]):
            raise ValueError("ERROR: Scaffold ssDNA overhangs must not be paired by staple nucleotides.")

    def set_nucleotide_model(self, nucleotide_model: HadoNucleotideModel | None, prepare_exports: bool = False):
        """Store the current nucleotide model, optionally preparing export-specific representations."""
        self.nucleotide_level_model = nucleotide_model
        if nucleotide_model is not None and prepare_exports:
            self._set_scadnano(nucleotide_model)
            self._set_oxdna(nucleotide_model)

    def get_nucleotide_model(self) -> HadoNucleotideModel | None:
        return self.nucleotide_level_model

    def set_lattice_type(self, lattice_type: str) -> None:
        """Switch scaffold lattice configuration and clear any lattice-specific nucleotide model."""
        lattice = get_lattice_config(lattice_type)
        self.scaffold_args.lattice_type = lattice.name
        self.nucleotide_level_model = None

    def set_custom_cross_section(self, custom_cross_section) -> None:
        """Use explicit forward/reverse helix coordinates for subsequent runs."""
        self.scaffold_args.set_custom_cross_section(custom_cross_section)
        self.nucleotide_level_model = None

    def set_cross_section_generator(self, cross_section_generator) -> None:
        """Use a callable custom cross-section generator for subsequent runs."""
        self.scaffold_args.set_cross_section_generator(cross_section_generator)
        self.nucleotide_level_model = None

    def clear_custom_cross_section(self) -> None:
        """Return to the registered lattice cross-section generator."""
        self.scaffold_args.clear_custom_cross_section()
        self.nucleotide_level_model = None

    def _resolve_nucleotide_model(
            self,
            nucleotide_model: HadoNucleotideModel | dict | None = None,
    ) -> HadoNucleotideModel:
        if nucleotide_model is None:
            nucleotide_model = self.nucleotide_level_model
        elif isinstance(nucleotide_model, dict):
            nucleotide_model = HadoNucleotideModel.load_from_dict(
                self.geometry,
                self.scaffold_args,
                self.staple_args,
                nucleotide_model,
            )

        if not isinstance(nucleotide_model, HadoNucleotideModel):
            raise ValueError(
                "ERROR: No nucleotide-level model is available. Run the manager or load a saved model first."
            )

        return nucleotide_model

    def get_estimate_num_nts(self, model: HadoNucleotideModel = None):
        """Estimate scaffold nucleotide count without always running the full pipeline."""
        model = self.nucleotide_level_model if model is None else model
        if model is not None and isinstance(model, HadoNucleotideModel):
            nts = model.get_scaffold_nucleotides()
            return np.sum(nts)
        else:
            axial_rise = get_lattice_config(self.scaffold_args.lattice_type).axial_rise
            if len(self.geometry.edges) < 12:
                # When low amount of edges, quickly just rotate and find approximate # of nts
                edge_xsect_definitions = perform_scaffold_routing(self.geometry, self.scaffold_args)
                design = initialize_base_design(self.geometry, self.scaffold_args, self.staple_args,
                                                edge_xsect_definitions)
                optimal_connections, best_state, scaffold_helix_connections = optimize_connections(design)
                final_positions, final_nts = miter_design(design, best_state, optimal_connections)
                return np.sum(final_nts)

            else:
                # Higher edges can take overly long for this, so:
                nts = 0
                for i, j in zip(self.geometry.edge_lengths_nm, self.geometry.n_per_edge):
                    nts += int((i * j) / axial_rise)
                return nts

    def to_json(self, nucleotide_level_model: HadoNucleotideModel = None, custom_oxview_dict: dict = None):
        """Serialize manager settings plus optional nucleotide model/custom oxView data."""
        nucleotide_level_model = self.nucleotide_level_model if nucleotide_level_model is None else nucleotide_level_model
        if nucleotide_level_model is not None:
            if isinstance(nucleotide_level_model, HadoNucleotideModel):
                nucleotide_level_model = nucleotide_level_model.to_dict()
            elif isinstance(nucleotide_level_model, dict):
                nucleotide_level_model = nucleotide_level_model

        data = {
            "design_name": self.design_name,
            "geometry": self.geometry.to_dict(),
            "scaffold_args": self.scaffold_args.to_dict(),
            "staple_args": self.staple_args.to_dict(),
            "verbose": self.verbose,
            'nucleotide_level_model': nucleotide_level_model,
            'custom_oxview': custom_oxview_dict,
        }
        return data

    def save(self, filename_no_extension: str = None, filepath: str | Path = ".",
             nucleotide_level_model: HadoNucleotideModel = None) -> dict:
        """ Serializes the manager state to a JSON (.hado) file. Returns the dictionary that was json dumped. """
        data = self.to_json(nucleotide_level_model)
        base = Path(filepath).expanduser() if filepath else Path.cwd()
        base = base.resolve()

        name = filename_no_extension or self.design_name
        name = Path(name).name

        fp = base / f"{name}.hado"
        fp.parent.mkdir(parents=True, exist_ok=True)

        with fp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        emit_runtime_message(f"HADO manager state saved to {filepath}", verbose=self.verbose)

        return data

    @classmethod
    def load(cls, filepath: Path | str = None, data: dict = None) -> 'HadoManager':
        """ Loads a .hado (which is a json) into the HadoManager object and returns it """
        if filepath is not None and data is not None:
            raise ValueError("ERROR: Can only load from one of filepath or data")

        if filepath is not None:
            with open(filepath, 'r') as f:
                data = json.load(f)

        fields = [
            'design_name',
            'geometry',
            'scaffold_args',
            'staple_args',
            'verbose',
        ]
        if not all(key in data for key in fields):
            raise ValueError(f"ERROR: Missing keys in HADO manager file. JSON file must contain these fields: {fields}")

        design_name = data['design_name']
        geo = Geometry(**data['geometry'])
        scaf = ScaffoldArgs(**data['scaffold_args'])
        stap = StapleArgs(**data['staple_args'])
        verbose = data['verbose']

        emit_runtime_message(f"HADO manager state loaded from {filepath}", verbose=verbose)

        if 'nucleotide_level_model' in data:
            nucleotide_level_model = HadoNucleotideModel.load_from_dict(geo, scaf, stap, data['nucleotide_level_model'])
        else:
            nucleotide_level_model = None
        manager = cls(design_name, geo, scaf, stap, verbose=verbose)
        manager.set_nucleotide_model(nucleotide_level_model)
        return manager

    @classmethod
    def load_default(cls, design_name: str):
        """
        Loads a default design from the app/pages/default_inputs directory
        Example usage: manager = HadoManager.load_default('tetrapod')
        """
        from hado.app.utils import DEFAULT_OPEN_MESHES, DEFAULT_SURFACE_MESHES
        clean_open = {k.lower(): v for k, v in DEFAULT_OPEN_MESHES.items()}
        clean_surface = {k.lower(): v for k, v in DEFAULT_SURFACE_MESHES.items()}
        if design_name.lower() in clean_open:
            filepath = clean_open[design_name]
        elif design_name.lower() in clean_surface:
            filepath = clean_surface[design_name]
        else:
            raise ValueError(f'ERROR: The specified design_name {design_name} was not found in the default directory.')
        return cls.load(filepath)

    @classmethod
    def load_ui(cls, filepath: Path | str = None, data: dict = None) -> tuple['HadoManager', HadoNucleotideModel, dict]:
        if filepath is not None and data is not None:
            raise ValueError("ERROR: Can only load from one of filepath or data")

        if filepath is not None:
            with open(filepath, 'r') as f:
                data = json.load(f)

        fields = [
            'design_name',
            'geometry',
            'scaffold_args',
            'staple_args',
            'verbose',
        ]
        if not all(key in data for key in fields):
            raise ValueError(f"ERROR: Missing keys in HADO manager file. JSON file must contain these fields: {fields}")

        design_name = data['design_name']
        geo = Geometry(**data['geometry'])
        scaf = ScaffoldArgs(**data['scaffold_args'])
        stap = StapleArgs(**data['staple_args'])
        verbose = data['verbose']

        emit_runtime_message(f"HADO manager state loaded from {filepath}", verbose=verbose)

        if 'nucleotide_level_model' in data:
            nucleotide_level_model = HadoNucleotideModel.load_from_dict(geo, scaf, stap, data['nucleotide_level_model'])
        else:
            nucleotide_level_model = None
        manager = cls(design_name, geo, scaf, stap, verbose=verbose)
        manager.set_nucleotide_model(nucleotide_level_model)
        autostaple_explorer = data['autostaple_explorer'] if 'autostaple_explorer' in data else {}

        return manager, nucleotide_level_model, autostaple_explorer

    def set_new_scaffold_sequence(self, sequence: str):
        """ Overwrite the scaffold sequence (e.g., to write a sequences file with a different scaffold) """
        sequence = sequence.upper()
        valid = set(sequence).issubset({'A', 'C', 'G', 'T'})
        if not valid:
            raise ValueError('ERROR: Invalid custom scaffold sequence, must contain A C G or T only.')
        self.scaffold_args.scaffold_sequence = sequence
        if self.nucleotide_level_model is not None:
            self.nucleotide_level_model._sc_design = None
            self.nucleotide_level_model._oxdna_system = None

    def write_sequences(self, nucleotide_model: HadoNucleotideModel = None, filepath: str | Path = ".",
                        filename_no_extension: str = None):
        """ Sequences the HadoNucleotideModel and exports a CSV """
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        fname = self.design_name if filename_no_extension is None else filename_no_extension
        worked, _ = sequence_design(
            nucleotide_model,
            self.scaffold_args,
            self.staple_args.unpaired_sequence,
            filepath,
            fname,
            self.verbose,
            True,
        )
        if not worked:
            raise ValueError('ERROR: Unable to write sequences, the number of scaffold nucleotides required is '
                             'too long for the scaffold')

    def get_sequences(self, nucleotide_model: HadoNucleotideModel = None) -> tuple[bool, list]:
        """Return scaffold/staple sequence rows without writing a CSV file."""
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        # Blank filepath / filename no extension because just returning sequences
        return sequence_design(
            nucleotide_model,
            self.scaffold_args,
            self.staple_args.unpaired_sequence,
            ".",
            self.design_name,
            self.verbose,
            False,
        )

    def write_cadnano(self, nucleotide_model: HadoNucleotideModel = None,
                      filepath: str | Path = ".", filename_no_extension: str = None):
        """ Write cadnano json file using the default design_name of the manager (unless filename_no_extension
        specified as an override) """
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        fname = self.design_name if filename_no_extension is None else filename_no_extension
        write_cadnano(fname, nucleotide_model, self.verbose, filepath=filepath)

    def write_scadnano(self, nucleotide_model: HadoNucleotideModel = None,
                       filepath: str | Path = ".", filename_no_extension: str = None):
        """ Write scadnano sc file using the default design_name of the manager (unless filename_no_extension
        specified as an override) """
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        fname = self.design_name if filename_no_extension is None else filename_no_extension
        self._set_scadnano(nucleotide_model)
        write_scadnano(fname, nucleotide_model, filepath=filepath)

    def write_oxdna(self, nucleotide_model: HadoNucleotideModel = None, use_old_top: bool = True,
                    filepath: str | Path = ".", filename_no_extension: str = None,
                    ):
        """ Write oxDNA top and dat files using the default design_name of the manager (unless filename_no_extension
        specified as an override). The use_old_top can eb set True / False for formatting purposes (if you have
        an old version of oxDNA installed, leave use_old_top=True.
        """
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        fname = self.design_name if filename_no_extension is None else filename_no_extension
        self._set_oxdna(nucleotide_model)
        write_oxDNA(
            fname,
            nucleotide_model,
            use_old_top,
            filepath=filepath,
            verbose=self.verbose,
        )

    def write_oxview(self, nucleotide_model: HadoNucleotideModel = None,
                     filepath: str | Path = ".", filename_no_extension: str = None):
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        fname = self.design_name if filename_no_extension is None else filename_no_extension
        self._set_oxdna(nucleotide_model)
        write_oxView(
            fname,
            nucleotide_model,
            filepath=filepath,
            verbose=self.verbose,
        )

    def get_cadnano_json(self, nucleotide_model: HadoNucleotideModel = None, filename_no_extension: str = None):
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        fname = self.design_name if filename_no_extension is None else filename_no_extension
        return get_cadnano_json(fname, nucleotide_model, self.verbose)

    def get_scadnano_json(self, nucleotide_model: HadoNucleotideModel = None):
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        self._set_scadnano(nucleotide_model)
        return get_scadnano_json(nucleotide_model)

    def get_oxview_json(self, nucleotide_model: HadoNucleotideModel = None):
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        self._set_oxdna(nucleotide_model)
        return get_oxview_json(nucleotide_model)

    def get_oxdna_strings(self, nucleotide_model: HadoNucleotideModel = None, use_old_top: bool = True):
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        self._set_oxdna(nucleotide_model)
        return get_oxDNA_strings(nucleotide_model, use_old_top, verbose=self.verbose)

    def get_oxview_json_and_oxdna_strings(self, nucleotide_model: HadoNucleotideModel = None,
                                          use_old_top: bool = True):
        """ This method is for easier grabbing of both strings for GUI export """
        nucleotide_model = self._resolve_nucleotide_model(nucleotide_model)
        self._set_oxdna(nucleotide_model)
        return get_oxView_json_and_oxDNA_Strings(
            nucleotide_model,
            use_old_top,
            verbose=self.verbose,
        )

    def _set_scadnano(self, nucleotide_model: HadoNucleotideModel):
        if nucleotide_model.scadnano_not_set():
            nucleotide_model.set_scadnano(
                self.scaffold_args.scaffold_sequence,
                self.staple_args.unpaired_sequence,
                verbose=self.verbose,
            )

    def _set_oxdna(self, nucleotide_model: HadoNucleotideModel):
        if nucleotide_model.oxdna_not_set():
            nucleotide_model.set_oxdna_system(self)

    @staticmethod
    def _run_mitering(design, state, optimal_connections, miter_distance, **kwargs) -> MiteringResult:
        return MiteringResult(
            *miter_design(design, state, optimal_connections, miter_distance, **kwargs)
        )

    def _run_mitering_with_half_turn_retry(self, design, best_state: dict, optimal_connections: dict,
                                           **kwargs) -> MiteringResult:
        """ The half-turn retry is specifically for small cross-sections (e.g., 2 or 3HB) where, due to geometry, the
        cross-sections may need a 180-degree rotation to helices are properly-constrained in length (num nts).
        Basically, the half-turn prevents a very sharp angle from arising due to the rotations in connect_bundles.
        """
        miter_distance = self.staple_args.target_miter_distance
        mitered = self._run_mitering(design, best_state, optimal_connections, miter_distance, **kwargs)
        is_valid, first_length = self._check_design_post_mitering(mitered.final_nts)
        if is_valid:
            return mitered

        # NOTE: Actually, below is a bad check, and really only generally negatively impacts the
        #       wireframe designs (that this package does not aim to necessarily reproduce)

        # rotated_state = {k: (v + 180) % 360 for k, v in best_state.items()}
        # rotated_mitered = self._run_mitering(design, rotated_state, optimal_connections, miter_distance, **kwargs)
        # is_valid, second_length = self._check_design_post_mitering(rotated_mitered.final_nts)
        # if is_valid:
        #     design.set_bundle_rotations(rotated_state)  # Update if this step used and validated
        #     return rotated_mitered
        # min_length = min(first_length, second_length)
        raise ValueError(
            f"ERROR: Edge post-mitering is too short ({first_length}) compared to "
            f"max threshold ({MIN_ALLOWABLE_EDGE_LENGTH_NTS}). Lengthen edge or reduce "
            f"cross section size."
        )

    @staticmethod
    def _check_design_post_mitering(check_final_nts: np.ndarray):
        """ Verifies that the edges all have AT LEAST the min number of nucleotides post-mitering """
        for fin in check_final_nts:
            temp = np.where(fin)[0]
            if len(temp) < MIN_ALLOWABLE_EDGE_LENGTH_NTS:
                return False, len(temp)

            is_contiguous = all(temp[i] == temp[i-1] + 1 for i in range(1, len(temp)))
            if not is_contiguous:
                return False, len(temp)

        return True, None
