"""Publication/integrity checks for the delivered Module 3 artifacts."""
import ast
import base64
import importlib.util
from pathlib import Path
import re
from urllib.parse import unquote

import nbformat

spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
h=runner.helpers();root=h['CFG'].root;ev=h['CFG'].evidence
checks=[]
def check(condition,label):
    assert condition,label
    checks.append(label)

def pointer(value,path):
    for p in path.strip('/').split('/') if path else []:
        value=value[int(p)] if isinstance(value,list) else value[p]
    return value

n=nbformat.read(root/'a03-diagnostic-loop.ipynb',as_version=4);nbformat.validate(n)
code=[c for c in n.cells if c.cell_type=='code']
check(len(code)==28 and [c.execution_count for c in code]==list(range(1,29)),'28 code cells executed in order')
check(not any(o.output_type=='error' for c in code for o in c.outputs),'No notebook error outputs')
check(all(i+1<len(n.cells) and n.cells[i+1].cell_type=='markdown' for i,c in enumerate(n.cells) if c.cell_type=='code'),'Every code cell immediately explained')
imports=[]
for i,c in enumerate(code):
    imports.extend((i,node) for node in ast.walk(ast.parse(c.source)) if isinstance(node,(ast.Import,ast.ImportFrom)))
check({i for i,node in imports}=={0},'One grouped imports cell; inline implementation')
check(not any('agentic_ts' in c.source for c in code),'No project-helper imports')
post=(root/'a03-post.md').read_text();outline=(root/'docs/module-3-post-outline.md').read_text()
nbheads=[line[3:] for c in n.cells if c.cell_type=='markdown' for line in c.source.splitlines() if line.startswith('## ')]
ph=re.findall(r'^## (.+)$',post,re.M);oh=re.findall(r'^## (.+)$',outline,re.M)
check(nbheads[:2]==['Setup','Utils'],'Setup and Utils first')
check(nbheads[2:]==ph==oh and ph[-1]=='Closing time','Exact shared substantive headings, hierarchy and order')
check('[ ✓ Subscribed ]' in post and 'United States of Banan is a reader-supported publication.' in post,'Publication footer present')

p=h['read_json'](ev/'current_run.json');directory=ev/'runs'/p['run_id'];b=h['verify_capture'](h['read_json'](directory/'capture.json'))
audit=h['verify_audit_gate'](b,directory);r=h['read_json'](directory/'results.json')
check(r['capture_hash']==b['bundle_hash'] and r['audit_hash']==h['digest'](audit),'Results bound to capture and pre-score author audit')
check(r['frozen_hash']==h['digest'](h['frozen_deliveries'](b)),'Frozen predictions bound to results')
check(audit['scored_outcomes_seen'] is False,'Author audit explicitly excludes final outcomes')
check(len(r['rows'])==9 and r['coverage']=={'A':3,'B':3,'C':3},'All9 deliveries/scored cases retained')
check(all(r['selection'][c]['full_menu_complete'] for c in b['cases']),'Completed comparator menus gate selection comparisons')
check(len(audit['steps'])==sum(len(c['C']['steps']) for c in b['cases'].values())==13,'Every primary request represented in author audit')
for case,arms in b['cases'].items():
    for arm,record in arms.items():
        check(record['timing']['total_seconds']>0,f'{case}/{arm} measured nonzero original duration')
    record=arms['C']
    check(record['work']['model_requests']==len(record['steps'])<=6,f'{case} request accounting')
    check(record['work']['tool_attempts']==sum('observation' in s for s in record['steps'])<=5,f'{case} tool accounting')
    for i,s in enumerate(record['steps'],1):
        check(s['response']['provider']['done'] and s['response']['provider']['done_reason']=='stop',f'{case}/{i} raw completed local inference retained')
        state=h['json'].loads(s['messages'][1]['content'])
        review=next(x for x in audit['steps'] if x['case']==case and x['step']==i)
        check(set(review['available_before'])==set(state['evidence']),f'{case}/{i} step-local audit scope')
        check(review['reason']==s['action']['reason'],f'{case}/{i} original public reason audited')
freeze=h['read_json'](ev/'protocol_freeze.json')
check(freeze['protocol_hash']==b['protocol_hash'],'Protocol frozen after development and before primary')
check(h['datetime'].fromisoformat(freeze['frozen_at'])<h['datetime'].fromisoformat(b['started'])<h['datetime'].fromisoformat(b['completed'])<h['datetime'].fromisoformat(audit['reviewed_at']),'Freeze/capture/audit chronological identities')
execution=h['read_json'](ev/'replay-execution.json')
check(execution['fresh_kernel'] and execution['network_blocked'] and execution['error_outputs']==0 and execution['code_cells']==28,'Full fresh-kernel numerical/decision replay with sockets blocked')
check(h['read_json'](ev/'development_checks.json')['passed'],'Behavioral pre-capture checks passed')

assets=h['read_json'](ev/'assets.json');qa=h['read_json'](ev/'visual_qa.json')
check(len(assets['assets'])==14 and [a['number'] for a in assets['assets']]==[f'{i:02d}' for i in range(1,15)],'14 assets in shared two-digit sequence')
postimages=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',post)
check(postimages==[a['filename'] for a in assets['assets']],'Post image order/filenames match asset manifest')
check(re.findall(r'!\[[^\]]*\]\(([^)]+)\)',outline)==postimages,'Outline images match final article')
inline=[base64.b64decode(o.data['image/png']) for c in code for o in c.outputs if o.output_type=='display_data' and 'image/png' in o.data]
check(len(inline)==14,'All14 publication PNGs displayed inline')
for i,a in enumerate(assets['assets']):
    path=root/a['filename'];side=root/a['sidecar'];value=h['read_json'](side)
    match=re.fullmatch(r'graphs/(graph|table)_(\d{2})_(\d{4})\.png',a['filename'])
    check(bool(match) and match[1]==a['type'] and match[2]==a['number'],f"Asset{a['number']} naming")
    check(match[3]==h['datetime'].fromisoformat(a['execution_timestamp']).strftime('%d%m'),f"Asset{a['number']} execution date")
    check(h['file_hash'](path)==a['png_sha256'] and h['file_hash'](side)==a['sidecar_sha256'] and h['digest'](value['source'])==a['source_sha256'],f"Asset{a['number']} source/sidecar/PNG integrity")
    check(inline[i]==path.read_bytes(),f"Asset{a['number']} inline/export bytes identical")
    reviewed=next(x for x in qa['assets'] if x['number']==a['number'])
    check(reviewed['passed'] and reviewed['png_sha256']==a['png_sha256'],f"Asset{a['number']} checksum-bound visual inspection")
    if a['type']=='table':
        rows=value['source']['rows'];formatted=value['display']
        for row,display in zip(rows,formatted):
            for key,text in zip(value['source']['columns'],display):
                raw=row[key]
                expected='N/A' if raw is None else f'{raw:.2f}' if isinstance(raw,float) else str(raw)
                check(text==expected,f"Asset{a['number']} typed display value")
check(assets['capture_hash']==b['bundle_hash'] and assets['results_hash']==h['digest'](r),'Asset manifest bound to original inference and scored results')
check(len(re.findall(r'^> \[Insert (?:Graph|Table) \d{2} here',post,re.M))==14,'All insertion markers present')
check(len(re.findall(r'\*\n\n[^\n]+',post))>=14,'Immediate readings follow visuals')
# All repository-relative references resolve; external URLs checked separately.
for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',post):
    if '://' not in target:
        check((root/unquote(target)).exists(),'Local article link resolves: '+target)
claim_record=h['read_json'](ev/'post_claims.json')
check(claim_record['post_sha256']==h['file_hash'](root/'a03-post.md'),'Claims bound to current article')
for claim in claim_record['claims']:
    path=root/claim['source_file']
    check(h['file_hash'](path)==claim['source_sha256'] and pointer(h['read_json'](path),claim['pointer'])==claim['value'],'Claim source: '+claim['id'])
    if claim['post_token']:
        check(claim['post_token'] in post,'Claim displayed token: '+claim['id'])
binding=h['read_json'](ev/'publication_bindings.json')
check(binding['post_sha256']==h['file_hash'](root/'a03-post.md') and binding['notebook_sha256']==h['file_hash'](root/'a03-diagnostic-loop.ipynb'),'Publication bindings current')
links=h['read_json'](ev/'link_validation.json')
check(all(x['verified'] for x in links['external']),'External research/prior-notebook links verified')
report={'passed':True,'checks':len(checks),'run_id':b['run_id'],'protocol_hash':b['protocol_hash'],
        'capture_hash':b['bundle_hash'],'results_hash':h['digest'](r),'post_sha256':h['file_hash'](root/'a03-post.md'),
        'notebook_sha256':h['file_hash'](root/'a03-diagnostic-loop.ipynb'),'asset_manifest_hash':h['digest'](assets),
        'fresh_kernel_execution':execution,'development_test_count':h['read_json'](ev/'development_checks.json')['tests'],
        'visuals_reviewed':14,'claims_bound':len(claim_record['claims']),
        'scope':'Archived first-attempt cohort, exact numerical/decision replay on recorded macOS arm64 environment; author explanation audit; no general-performance claim'}
h['save_json'](ev/'validation.json',report)
print(h['json'].dumps(report,indent=2))
