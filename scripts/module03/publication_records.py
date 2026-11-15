"""Bind the article/outline to the archived evidence and current assets."""
import importlib.util
from pathlib import Path
import re

spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
h=runner.helpers();root=h['CFG'].root;ev=h['CFG'].evidence
pointer=h['read_json'](ev/'current_run.json');directory=ev/'runs'/pointer['run_id']
b=h['verify_capture'](h['read_json'](directory/'capture.json'))
r=h['read_json'](directory/'results.json');audit=h['verify_audit_gate'](b,directory)
a=h['read_json'](ev/'assets.json');post=(root/'a03-post.md').read_text()

counts={}
for arm in ['A','B','C']:
 records=[b['cases'][c][arm] for c in ['C1','C2','C3']]
 counts[arm]={'tools':sum(x['work'].get('tool_attempts',0) for x in records),
              'fits':sum(x['work'].get('validation_fit_attempts',0)+x['work'].get('current_fit_attempts',0) for x in records),
              'requests':sum(x['work'].get('model_requests',0) for x in records),
              'seconds':sum(x['timing']['total_seconds'] for x in records)}
sequences={c:[s['action']['action']+':'+str(s['action']['arguments']) for s in b['cases'][c]['C']['steps']] for c in b['cases']}
lookup={(x['case'],x['arm']):x for x in r['rows']}
findings={'run_id':b['run_id'],'work':counts,'sequences':sequences,
          'distinct_sequences':len({tuple(x) for x in sequences.values()}),
          'c_vs_b_rmse':{'wins':sum(lookup[(c,'C')]['rmse']<lookup[(c,'B')]['rmse'] for c in b['cases']),
                         'ties':sum(lookup[(c,'C')]['rmse']==lookup[(c,'B')]['rmse'] for c in b['cases'])},
          'a_beats_both_rmse':sum(lookup[(c,'A')]['rmse']<min(lookup[(c,'B')]['rmse'],lookup[(c,'C')]['rmse']) for c in b['cases']),
          'audit':{key:sum(s[key] for s in audit['summary'].values()) for key in ['supported','factual_total','observation_used','observation_eligible','proposals','unassessable']},
          'audit_categories':{key:sum(s['categories'][key] for s in audit['summary'].values()) for key in ['supported','contradicted','unsupported','ambiguous']},
          'tokens':{key:sum(s['response']['usage'][key] for c in b['cases'].values() for s in c['C']['steps']) for key in ['input_tokens','output_tokens']},
          'all_deliveries':len(r['rows']),'fallbacks':sum(x['delivery_status']=='fallback' for x in r['rows']),
          'all_controller_complete':sum(c['C']['terminal']['status']=='completed' for c in b['cases'].values())}
h['save_json'](ev/'derived_findings.json',findings)

claims=[]
def claim(key,description,path,pointer,display=None,token=None):
 value=h['read_json'](path)
 for part in pointer.strip('/').split('/') if pointer else []:
  value=value[int(part)] if isinstance(value,list) else value[part]
 claims.append({'id':key,'claim':description,'source_file':str(path.relative_to(root)),
                'source_sha256':h['file_hash'](path),'pointer':pointer,'value':value,'display':display,
                'post_token':token,'run_id':b['run_id'],'protocol_hash':b['protocol_hash']})
for i,row in enumerate(r['rows']):
 for metric in ['rmse','mae','mase']:
  value=row[metric]
  claim(f"{row['case']}-{row['arm']}-{metric}",f"Table11 {row['case']}/{row['arm']} {metric}",directory/'results.json',f'/rows/{i}/{metric}',f'{value:.2f}')
for case in ['C1','C2','C3']:
 claim(case+'-regret','Historical validation selection gap',directory/'results.json',f'/selection/{case}/regret',f"{r['selection'][case]['regret']:.2f}")
 claim(case+'-omitted','Better omitted validation candidate(s)',directory/'results.json',f'/selection/{case}/omitted_better')
for arm in ['B','C']:
 for key in ['tools','fits','requests','seconds']:
  val=counts[arm][key]
  claim(f'work-{arm}-{key}','Summed observed primary work/durations',ev/'derived_findings.json',f'/work/{arm}/{key}',f'{val:.2f}' if isinstance(val,float) else str(val),f'{val:.2f}' if key=='seconds' else None)
for key in ['distinct_sequences','a_beats_both_rmse','all_deliveries','fallbacks','all_controller_complete']:
 claim(key,'Declared cohort outcome count',ev/'derived_findings.json','/'+key)
for key in ['wins','ties']:
 claim('c-vs-b-'+key,'Full-precision paired final RMSE comparison',ev/'derived_findings.json','/c_vs_b_rmse/'+key)
for key in findings['audit']:
 claim('audit-'+key,'Complete author audit denominator/count',ev/'derived_findings.json','/audit/'+key)
for key in findings['audit_categories']:
 claim('audit-category-'+key,'Author atomic factual classification',ev/'derived_findings.json','/audit_categories/'+key)
for case in ['C1','C2','C3']:
 for candidate in ['seasonal_naive','hw_full']:
  val=b['cases'][case]['C']['tested_candidates'][candidate]['mean_rmse']
  claim(f'cv-{case}-{candidate}','Pre-outcome three-fold mean validation RMSE',directory/'capture.json',f'/cases/{case}/C/tested_candidates/{candidate}/mean_rmse',f'{val:.2f}')
for candidate in ['seasonal_naive','hw_full']:
 val=b['cases']['C2']['C']['tested_candidates'][candidate]['folds'][0]['metrics']['rmse']
 claim(f'c2-first-fold-{candidate}','First-fold counterexample to all-folds claim',directory/'capture.json',f'/cases/C2/C/tested_candidates/{candidate}/folds/0/metrics/rmse',f'{val:.2f}',f'{val:.2f}')
# The complete per-observation source supports all diagnostic numbers in trace images/prose.
for case in ['C1','C2','C3']:
 claim('observations-'+case,'Every diagnostic value and original reason in the trace',directory/'capture.json',f'/cases/{case}/C/steps')
scripted=ev/'exports'/a['export_id']/'scripted_examples.json'
claim('scripted-stimuli','Synthetic strengths/growth/residual measurements; not genuine-agent evidence',scripted,'')
freeze=h['read_json'](ev/'protocol_freeze.json');pilot=root/freeze['pilot_path']
val=h['read_json'](pilot)['seconds']
claim('preflight-seconds','Separate synthetic preflight end-to-end duration',pilot,'/seconds',f'{val:.2f}',f'{val:.2f}')
claim('study-protocol','Cases, boundaries, windows, budgets, settings, provenance and interpretation rules',ev/'protocol.json','')
claims_record={'run_id':b['run_id'],'protocol_hash':b['protocol_hash'],'capture_hash':b['bundle_hash'],
               'results_hash':h['digest'](r),'post_sha256':h['file_hash'](root/'a03-post.md'),'claims':claims,
               'qualitative_review':[
                   {'claim':'Two distinct genuine sequences; cafes adds growth; none tests recent60','evidence':'derived_findings.sequences + all capture steps'},
                   {'claim':'STL results often not substantively used; identical sequences do not establish causal adaptation','evidence':'decision_audit.steps, rubric and complete raw reasons'},
                   {'claim':'Adequacy claims exceed descriptive tool outputs','evidence':'source tool definitions and reviewed unsupported/ambiguous propositions'},
                   {'claim':'Fewer fits, much longer measured duration; no broad speed or forecasting-superiority claim','evidence':'derived_findings.work and results.rows; selected-case one-origin scope'},
               ],
               'background_sources':[
                   {'url':'https://arxiv.org/abs/2210.03629','claim':'ReAct interleaves reasoning/action/environment observations in evaluated language/interactive tasks'},
                   {'url':'https://www.anthropic.com/research/building-effective-agents','claim':'Workflow versus runtime agent control, latency/cost tradeoff'},
                   {'url':'https://otexts.com/fpp3/toolbox.html','claim':'Fitted residuals, forecast error, temporal evaluation and scaled error concepts'},
               ]}
h['save_json'](ev/'post_claims.json',claims_record)

# Refresh the outline with exact visual blocks and their immediate readings.
purposes=[
 'Define runtime control, source lineage and continuity with m01/m03/m16 and post1.',
 'Establish units, chosen questions, permitted histories and retrospective information boundaries.',
 'Explain seasonal naive, two HW windows, chronological validation and the matched fixed control.',
 'Define action/state/observation contracts and budgets; distinguish scripted mechanics from live evidence.',
 'Walk every genuine first attempt, exposing unused observations and unsupported or false reasons.',
 'Open audited frozen forecasts; compare all errors, coverage, actual work and explanation support.',
 'Demonstrate deterministic limits/failures without estimating live failure rates.',
 'Answer the opening question with mixed findings and conditions for considering the loop.',
]
parts=re.split(r'(?m)^## ',post)[1:]
outline=['# Module 3 post outline — executed evidence','','Opening: practical diagnostic questions, runtime choice and a preview of mixed observed outcomes. Companion notebook link verified locally.','']
for index,part in enumerate(parts):
 title=part.splitlines()[0];text='\n'.join(part.splitlines()[1:])
 outline += ['## '+title,'','**Purpose:** '+purposes[index], '', '**Evidence:** frozen protocol '+b['protocol_hash']+'; original inference '+b['run_id']+'.','']
 blocks=re.findall(r'(> \[Insert (?:Graph|Table).*?\n\n!\[.*?\n\n\*.*?\n\n[^\n]+)',text,re.S)
 for block in blocks:
  outline += [block.strip(),'']
 outline += ['**Reader takeaway:** '+purposes[index],'']
(root/'docs/module-3-post-outline.md').write_text('\n'.join(outline)+'\n')
# Keep exact manuscript captions/alt text next to the inference asset manifest.
bindings=[]
for asset in a['assets']:
 path=re.escape(asset['filename'])
 match=re.search(r'!\[([^\]]+)\]\('+path+r'\)\n\n\*([^*]+)\*\n\n([^\n]+)',post)
 assert match,asset['filename']
 bindings.append({'number':asset['number'],'filename':asset['filename'],'png_sha256':asset['png_sha256'],
                  'post_alt':match[1],'post_caption':match[2],'immediate_reading':match[3],
                  'notebook_manifest_caption':asset['caption']})
h['save_json'](ev/'publication_bindings.json',{'post_sha256':h['file_hash'](root/'a03-post.md'),
 'notebook_sha256':h['file_hash'](root/'a03-diagnostic-loop.ipynb'),'asset_manifest_hash':h['digest'](a),'bindings':bindings})
print('Claims:',len(claims),'words:',len(post.split()),'outline sections:',len(parts),'bindings:',len(bindings))
