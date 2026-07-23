from hado import HadoManager
from pathlib import Path

all_inputs = r'../../hado/app/pages/default_inputs'


for fp in Path(all_inputs).glob('*.json'):
    try:
        manager = HadoManager.load(fp)
        model = manager.run()
        manager.write_oxview(filepath='oxdna_files')
        print(f'SUCCESS: {fp.stem}')
    except Exception as e:
        print(f'ERROR for design {fp.stem}: {e}')
