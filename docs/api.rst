API Reference
=============

These are the main entry points for typical use. The `HadoManager` is the main data structure that will run the automation pipeline and create the `HadoNucleotideDesign` structure that can then be exported to other CAD software (e.g., oxDNA, caDNAno, scadnano). Generally, there is not much to read here because the package is meant to automate a coarse input (i.e., a mesh) and maybe some custom args!

However, the `HadoManager` required the `Geometry`, `StapleArgs`, and `ScaffoldArgs` as inputs. Overall, the default values can / should be used in `StapleArgs` as these are what were used in experimental characterization in the journal article. However, the `scaffold_sequence` in the `ScaffoldArgs` can be used to ensure staples are properly sequenced. The Geometry class can be used to read-in standard mesh files (e.g., .ply or .obj) OR a list of vertices and edges can be passed in to create the `Geometry`. Once these three args are defined, you can pass them into the `HadoManager` and run the automated design algorithm using `HadoManager.run()`. From there, you can export this final model to other tools by e.g., `HadoManager.write_cadnano()`.

Please see the Tutorial notebook for a walk-through on how to use this package.

Note that the API documentation has been auto-generated using Sphinx, if you have specific questions please reach out or open a GitHub Issue and I will try and help out as much as I can.

.. autofunction:: hado.HadoManager
.. autoclass:: hado.ScaffoldArgs
.. autoclass:: hado.StapleArgs
.. autoclass:: hado.Geometry
.. autoclass:: hado.HadoNucleotideModel
