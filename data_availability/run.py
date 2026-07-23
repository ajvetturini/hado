from hado import HadoManager

# This will use the default honeycomb-style grid upon loading
honeycomb = HadoManager.load_default('plus sign')
honeycomb.run()
honeycomb.write_oxView(filename_no_extension='honeycomb_plus')

# You can also set a square grid:
square = HadoManager.load_default('plus sign')
square.set_lattice_type('dna_square')
square.run()
square.write_oxView(filename_no_extension='square_plus')
