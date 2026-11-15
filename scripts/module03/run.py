"""Developer entry points; runtime implementation is read from notebook cells."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/module03-matplotlib')


def helpers():
    import nbformat
    n = nbformat.read(ROOT / 'a03-diagnostic-loop.ipynb', as_version=4)
    ns = {}
    for i, cell in enumerate(n.cells):
        if cell.cell_type == 'markdown' and cell.source.startswith('## Why let'):
            break
        if cell.cell_type == 'code':
            exec(compile(cell.source, f'a03-cell-{i}', 'exec'), ns)
    return ns


def pilot_and_freeze():
    h = helpers()
    cfg = h['CFG']
    stamp = datetime.now(ZoneInfo(cfg.timezone)).strftime('pilot-%Y%m%dT%H%M%S%f%z')
    directory = cfg.evidence / 'development' / stamp
    directory.mkdir(parents=True, exist_ok=False)
    identity = h['validate_endpoint']()
    protocol = h['protocol_value'](identity)
    session = h['DiagnosticSession'](h['synthetic_prefixes']()['seasonal'], {'series_id': 'synthetic-preflight'})
    state = h['state_for'](session, [], protocol)
    messages = [{'role': 'system', 'content': h['SYSTEM']}, {'role': 'user', 'content': h['json_text'](state)}]
    started = time.perf_counter()
    result = {'purpose': 'synthetic transport/schema preflight; outside primary denominators', 'identity': identity,
              'messages': messages, 'settings': h['model_settings']()}
    try:
        response = h['ollama_request'](messages, cfg.request_seconds)
        result['response'] = response
        result['action'] = h['validate_action'](h['strict_json'](response['text']), session)
        if response['provider'].get('done_reason') == 'length':
            raise ValueError('Pilot hit output cap')
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    result['seconds'] = time.perf_counter()-started
    h['save_json'](directory / 'pilot.json', result)
    print('Pilot:', result['status'], f"{result['seconds']:.2f}s", flush=True)
    if result['status'] != 'passed':
        raise RuntimeError(result['error'])
    development = h['read_json'](cfg.evidence / 'development_checks.json')
    qa = h['read_json'](cfg.evidence / 'development' / 'prototype_review.json')
    assert development['passed'] and qa['passed'], 'Development tests/prototype QA must precede freeze'
    h['save_json'](cfg.evidence / 'protocol.json', protocol)
    h['save_json'](cfg.evidence / 'protocol_freeze.json', {
        'protocol_hash': h['digest'](protocol), 'frozen_at': datetime.now(ZoneInfo(cfg.timezone)).isoformat(),
        'development_checks_hash': h['digest'](development), 'prototype_review_hash': h['digest'](qa),
        'pilot_path': str((directory/'pilot.json').relative_to(ROOT)), 'pilot_hash': h['digest'](result),
        'primary_data_not_scored': True})
    print('Frozen protocol:', h['digest'](protocol), flush=True)


def execute(mode):
    import nbformat
    from nbclient import NotebookClient
    os.environ['A03_MODE'] = 'live' if mode == 'capture' else 'replay'
    path = ROOT / 'a03-diagnostic-loop.ipynb'
    n = nbformat.read(path, as_version=4)
    if mode == 'capture':
        end = next(i for i,c in enumerate(n.cells) if 'capture_finished' in c.metadata.get('tags', []))
        n.cells = n.cells[:end+1]
    else:
        # Patch at notebook runtime, after kernel startup; no model/network access permitted.
        block = nbformat.v4.new_code_cell('''import socket as _offline_socket

def _reject_network(*args, **kwargs):
    raise RuntimeError("Network/model access blocked during verified replay")

_offline_socket.create_connection = _reject_network
_offline_socket.getaddrinfo = _reject_network
_offline_socket.socket.connect = _reject_network
''', metadata={'tags': ['offline_verification']})
        n.cells.insert(0, block)
    started = time.perf_counter()
    client = NotebookClient(n, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    except BaseException:
        nbformat.write(n, ROOT / 'examples/module03' / f'{mode}-interrupted.ipynb')
        raise
    elapsed = time.perf_counter()-started
    if mode == 'capture':
        target = ROOT / 'examples/module03' / 'primary-executed-prefix.ipynb'
    else:
        n.cells = [c for c in n.cells if 'offline_verification' not in c.metadata.get('tags', [])]
        # Preserve original cell numbers; the verification-only network guard is not a teaching cell.
        for count, c in enumerate((c for c in n.cells if c.cell_type=='code'), start=1):
            c.execution_count = count
        target = path
    nbformat.write(n, target)
    record = {'mode': mode, 'fresh_kernel': True, 'seconds': elapsed,
              'network_blocked': mode != 'capture', 'code_cells': sum(c.cell_type=='code' for c in n.cells),
              'error_outputs': sum(o.output_type=='error' for c in n.cells if c.cell_type=='code' for o in c.get('outputs', [])),
              'target': str(target.relative_to(ROOT))}
    (ROOT/'examples/module03'/f'{mode}-execution.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


def prototypes():
    h = helpers()
    cfg = h['CFG']
    cfg.graphs = cfg.evidence / 'development' / 'prototypes'
    cfg.graphs.mkdir(parents=True, exist_ok=True)
    cfg.table_wrap = 95
    h['STUDY'] = {'run_id': 'synthetic-layout-prototype'}
    sample = [{'Step / action': f'{i}. benchmark hw_recent60',
               'Returned evidence': f'o{i}: recent60; eligible True; CV RMSE123.45',
               'Public reason (verbatim)': 'The recent growth contrast is large relative to historical seasonal differences. Test the shorter history and compare chronological validation before deciding whether it helps. [o1, baseline]',
               'Status': 'ok'} for i in range(1,7)]
    sample.append({'Step / action':'Terminal', 'Returned evidence':'agent_stop', 'Public reason (verbatim)':'Python delivery: hw_recent60', 'Status':'complete'})
    h['table_visual'](5, sample, 'Development prototype', 'Synthetic layout, not evidence', 'Synthetic longest trace prototype', widths=cfg.trace_widths)
    rows, records = h['failure_fixtures']({'development': True})
    h['table_visual'](14, rows, 'Development prototype', 'Synthetic layout, not evidence', 'Failure table prototype', widths=[.25,.29,.13,.12,.21])
    h['save_json'](cfg.evidence/'development'/'prototype_manifest.json', h['ASSETS'])
    print('\n'.join(str(cfg.graphs/Path(a['filename']).name) for a in h['ASSETS']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['pilot-freeze','capture','replay','prototypes'])
    args = parser.parse_args()
    if args.command == 'pilot-freeze':
        pilot_and_freeze()
    elif args.command == 'prototypes':
        prototypes()
    else:
        execute(args.command)
