import os
from hado import HadoManager

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.hado'):
        f = os.path.join(os.path.dirname(__file__), file)
        try:
            manager = HadoManager.load(f)
        except Exception as e:
            print(f'ERROR reading in {file}: {e}')
            continue

        # manager.staple_args.target_miter_distance = 2.0
        # manager.save()


