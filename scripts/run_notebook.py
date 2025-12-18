from pathlib import Path
import nbformat
from nbclient import NotebookClient

nb_path = Path('notebooks') / '04_Sales_Agent.ipynb'
if not nb_path.exists():
    raise SystemExit(f'Notebook not found: {nb_path}')

nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='python3')
print('Executing notebook:', nb_path)
client.execute()
out_path = nb_path.parent / '04_Sales_Agent.executed.ipynb'
nbformat.write(nb, out_path)
print('Wrote executed notebook:', out_path)
