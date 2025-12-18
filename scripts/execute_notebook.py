import nbformat
from nbclient import NotebookClient
from pathlib import Path

NB_IN = Path('notebooks/03_Core_State_Models.ipynb')
NB_OUT = NB_IN

nb = nbformat.read(str(NB_IN), as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()
cnt = 0
for i, cell in enumerate(nb.cells):
	if cell.get('cell_type') == 'code':
		outputs = cell.get('outputs', [])
		print(f'Cell', i+1, 'outputs=', len(outputs))
		if outputs:
			cnt += 1
			# print a short preview
			out = outputs[0]
			if out.get('output_type') == 'stream':
				text = out.get('text')
				print('  preview:', text.splitlines()[:5])

nbformat.write(nb, str(NB_OUT))
print('Overwrote', NB_OUT, 'with executed outputs (', cnt, 'cells contained outputs)')

# Also create a copy where outputs are embedded as markdown cells after each code cell
from nbformat import v4 as nbfv4
new_nb = nbfv4.new_notebook(metadata=nb.metadata)
new_cells = []
for cell in nb.cells:
	new_cells.append(cell)
	if cell.get('cell_type') == 'code':
		outputs = cell.get('outputs', [])
		if outputs:
			lines = []
			for out in outputs:
				if out.get('output_type') == 'stream':
					lines.extend(out.get('text','').splitlines())
				elif out.get('output_type') == 'execute_result' or out.get('output_type') == 'display_data':
					# try to extract text/plain
					data = out.get('data', {})
					text = data.get('text/plain') or data.get('text') or ''
					if isinstance(text, list):
						lines.extend(text)
					else:
						lines.extend(str(text).splitlines())
				elif out.get('output_type') == 'error':
					lines.append('\n'.join(out.get('traceback', [])))
			if lines:
				md = nbfv4.new_markdown_cell('\n'.join(['**Cell output:**'] + [''] + lines))
				new_cells.append(md)
new_nb['cells'] = new_cells
OUT2 = Path('notebooks/03_Core_State_Models.with_outputs.ipynb')
nbformat.write(new_nb, str(OUT2))
print('Wrote notebook with embedded outputs ->', OUT2)
