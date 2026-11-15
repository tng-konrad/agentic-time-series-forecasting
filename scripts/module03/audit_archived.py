"""Serialize the author's step-local review of this one archived, unscored cohort.

This is a manual-review transcription, not an automated truth judge. Refuses new
run identities. Proposals are excluded from factual denominators; partial valid
observation use does not imply the whole reason is sound or the action optimal.
"""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
h=runner.helpers()
RUN='live-20261112T123814174366+0100'
CAPTURE='0f82a8d4463d9e8c4129ef57f46d718e2066dc853f81a75f326b9e4b5127f6c6'
directory=h['CFG'].evidence/'runs'/RUN
bundle=h['verify_capture'](h['read_json'](directory/'capture.json'))
assert bundle['run_id']==RUN and bundle['bundle_hash']==CAPTURE, 'A new run needs a new actual author review'
assert not (directory/'results.json').exists(), 'This audit must be completed before opening final outcomes'

# Atomic comparative proposition includes its two stated operand values; it is
# one assessed relation, not three duplicated claims. Separate residual values,
# adequacy verdicts and operational claims are independently assessable.
FACTS={
 ('C1',4):[
  ('hw_full outperforms seasonal_naive (mean RMSE87.67 vs133.83)','supported',['baseline','o2'],'Means87.66545<133.82962; stated2decimal values agree.'),
  ('ACF1=0.30','supported',['o3'],'Returned0.302001; rounding agrees.'),
  ('ACF12=0.11','supported',['o3'],'Returned0.113723; rounding agrees.'),
  ('residuals show no severe autocorrelation','ambiguous',['o3'],'Severe has no defined threshold or calibrated test; numeric correlations alone do not settle it.'),
  ('adequate for forecasting','unsupported',['o3'],'No adequacy verdict/test is returned. Descriptive fitted residuals cannot certify twelve-step forecast quality.'),
 ],
 ('C2',4):[
  ('hw_full outperformed seasonal_naive in all folds','contradicted',['baseline','o3'],'Fold1 hw_full230.26121 exceeds baseline198.08757. Mean improvement is not every-fold improvement.'),
 ],
 ('C2',5):[
  ('hw_full outperformed seasonal_naive in all folds','contradicted',['baseline','o3'],'Same fold1 counterexample; repeated occurrence counted separately.'),
  ('mean RMSE130.1 vs227.6','supported',['baseline','o3'],'Means130.14920 and227.59124 agree with the stated1decimal comparison.'),
  ('residuals show no concerning autocorrelation','ambiguous',['o4'],'Concerning has no predeclared threshold or calibrated test.'),
  ('lag12 ACF approximately0','supported',['o4'],'Returned0.008756 is descriptively near0; this does not prove independence.'),
  ('budget exhausted','contradicted',[],'Before this stop, state has2requests and1tool remaining; investigation15.74seconds is well below300. No token-limit termination.'),
 ],
 ('C3',4):[
  ('hw_full outperforms seasonal_naive (mean RMSE116.9 vs175.1)','supported',['baseline','o2'],'Means116.93318<175.10788; stated1decimal values agree.'),
  ('residuals show mild autocorrelation','ambiguous',['o3'],'ACF1=.278866 andACF12=.247724; mild is undefined and no severity calibration exists.'),
  ('no diagnostics flag issues','unsupported',['o1','o2','o3'],'Tools return descriptions and execution statuses, not an adequacy flag. Successful execution is not a passed diagnostic test.'),
  ('forecast ready','ambiguous',['o2','o3'],'Operational finalization is permitted; acceptable forecast quality is not established. Wording conflates these readings.'),
 ],
}
USE={
 ('C1',2):(False,'Cites baseline only; ignores latest STL result.'),
 ('C1',3):(False,'Cites latest benchmark but reason only proposes a residual check; it states no returned result or result-to-action relationship.'),
 ('C1',4):(True,'Uses latest residualACF values explicitly in stop rationale. Those numerical facts are supported; the separate severity/adequacy claims are not established.'),
 ('C2',2):(False,'Cites baseline only; no use of latest STL result.'),
 ('C2',3):(False,'Cites latest growth observation but gives a generic benchmark proposal without relating returned growth to candidate/window choice.'),
 ('C2',4):(False,'Latest benchmark is cited, but the substantive all-folds assertion is contradicted.'),
 ('C2',5):(True,'Uses latest nearzero lag12 residualACF in stop rationale; records partial supported use only. Other claims remain contradicted/ambiguous.'),
 ('C3',2):(False,'Cites baseline only; latest STL observation is unused.'),
 ('C3',3):(False,'Generic residual-check proposal cites benchmark, but no substantive returned-result relationship is stated.'),
 ('C3',4):(False,'Latest residuals cited, but qualitative mild/no-issues adequacy rationale is ambiguous or unsupported; no clear supported relationship.'),
}
steps=[];propositions=[];summary={}
for case in ['C1','C2','C3']:
 record=bundle['cases'][case]['C'];known={'baseline','request','protocol'};latest=None
 for i,step in enumerate(record['steps'],1):
  action=step['action'];facts=FACTS.get((case,i),[])
  eligible=latest is not None
  use,why=USE.get((case,i),(False,'First request; no returned tool observation yet.'))
  proposals=[]
  if not facts:
   proposals=[{'text':action['reason'],'classification':'proposal','note':'Proposed measurement/benchmark, not an assertion that its hypothesized condition holds.'}]
  elif (case,i)==('C2',4):
   proposals=[{'text':'check residuals for final candidate to confirm adequacy','classification':'proposal','note':'Checking residuals is proposed; the adequacy aim exceeds what this descriptive tool can certify.'}]
  ids=[]
  for j,(text,category,refs,note) in enumerate(facts,1):
   assert set(refs)<=known
   claim_id=f'{case}-s{i}-p{j}';ids.append(claim_id)
   propositions.append({'id':claim_id,'case':case,'step':i,'text':text,'category':category,'evidence_ids':refs,'justification':note,'reviewer':'Codex author'})
  steps.append({'case':case,'step':i,'reason':action['reason'],'available_before':sorted(known),
                'cited_ids':action['evidence_ids'],'latest_observation':latest,
                'proposition_ids':ids,'proposals':proposals,'observation_eligible':eligible,
                'observation_used':use if eligible else False,'observation_justification':why,'unassessable':False})
  if 'observation' in step:
   latest=step['observation']['id'];known.add(latest)
 p=[x for x in propositions if x['case']==case];s=[x for x in steps if x['case']==case]
 summary[case]={'supported':sum(x['category']=='supported' for x in p),'factual_total':len(p),
                'categories':{c:sum(x['category']==c for x in p) for c in ['supported','contradicted','unsupported','ambiguous']},
                'observation_used':sum(x['observation_used'] for x in s),'observation_eligible':sum(x['observation_eligible'] for x in s),
                'proposals':sum(len(x['proposals']) for x in s),'unassessable':sum(x['unassessable'] for x in s)}
audit={'complete':True,'reviewer':'Codex (author audit; not independent quality evaluation)',
       'reviewed_at':h['datetime'].now(h['ZoneInfo'](h['CFG'].timezone)).isoformat(),
       'capture_hash':CAPTURE,'frozen_hash':h['digest'](h['frozen_deliveries'](bundle)),
       'run_id':RUN,'protocol_hash':bundle['protocol_hash'],
       'rubric':bundle['protocol']['audit'],
       'unit_note':'A numeric comparative relation includes its operands as one proposition. Compound severity/adequacy assertions split. Repeated claims in different steps count separately.',
       'observation_use_note':'Conservative author assessment: at least one explicit supported result-to-action link using latest observation; generic proposals/citations alone excluded. Partial support can count even if other reason clauses fail; not optimality or causal adaptation.',
       'steps':steps,'propositions':propositions,'summary':summary,'scored_outcomes_seen':False}
h['save_json'](directory/'decision_audit.json',audit)
lines=['# Pre-score author decision audit','','Reviewer: Codex author; step-local evidence only. Not an independent quality estimate.','',f'Run `{RUN}`; capture `{CAPTURE}`.','']
for case,value in summary.items():
 lines += [f'## {case}', '',str(value),'']
 for p in propositions:
  if p['case']==case:lines += [f"- {p['id']}: **{p['category']}** — {p['text']}. {p['justification']}"]
 lines.append('')
lines += ['Pure diagnostic proposals are excluded from factual denominators. The observation-use count records stated supported use, not whether evidence causally changed the decision. Full per-step exclusions and notes are in decision_audit.json.']
(directory/'decision_audit.md').write_text('\n'.join(lines)+'\n')
print(summary)
