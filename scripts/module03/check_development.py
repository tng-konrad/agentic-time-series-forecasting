"""Behavioral pre-capture checks against the inline notebook implementation."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
h = runner.helpers()
np, pd = h['np'], h['pd']


class Checks(unittest.TestCase):
    def setUp(self):
        self.prefix = h['synthetic_prefixes']()['seasonal']
        self.meta = {'series_id':'fixture'}
        self.protocol = {'development': True}
    def session(self):
        return h['DiagnosticSession'](self.prefix, self.meta, fitter=h['fixture_fit'])
    def run_actions(self, actions):
        return h['run_adaptive'](self.session(), self.protocol, client=h['scripted_client'](actions))
    def test_metric_arithmetic_and_undefined_scale(self):
        result = h['metrics']([2,4], [1,2], list(range(24)))
        self.assertEqual(result['mae'], 1.5)
        self.assertEqual(result['rmse'], np.sqrt(2.5))
        self.assertEqual(result['mase'], 1.5/12)
        s = h['DiagnosticSession'](pd.Series(100.,index=self.prefix.index), self.meta, fitter=h['fixture_fit'])
        self.assertTrue(s.baseline['eligible'])
        self.assertIsNone(s.baseline['folds'][0]['metrics']['mase'])
        self.assertEqual(s.finalize()['forecast_status'], 'ok')
    def test_chronological_windows_and_selection(self):
        s = self.session()
        b = s.benchmark('hw_recent60')
        self.assertEqual([f['fit_start'] for f in b['folds']], ['2016-07-01','2017-07-01','2018-07-01'])
        self.assertEqual([f['available_n'] for f in b['folds']], [258,270,282])
        self.assertEqual(h['select_candidate'](s.tested), 'seasonal_naive')
        fitted = h['fit_current'](self.prefix, 'hw_recent60')
        self.assertEqual(fitted['train_start'],'2019-07-01')
        self.assertEqual(len(fitted['forecast']),12)
    def test_shift_statistic_and_limitation(self):
        for onset, expected in [(270,15.),(264,0.)]:
            values = np.ones(294)*100 + np.where(np.arange(294)>=onset,30,0)
            s = h['DiagnosticSession'](pd.Series(values,index=self.prefix.index), self.meta, fitter=h['fixture_fit'])
            self.assertEqual(s.compare_recent_history()['contrast'],expected)
        s = self.session()
        annual = self.prefix.diff(12)
        self.assertEqual(s.compare_recent_history()['contrast'],float(annual.iloc[-24:].median()-annual.iloc[-48:-24].median()))
    def test_residual_alignment_and_constant(self):
        s = self.session()
        actual = s.check_residuals('seasonal_naive')
        expected = h['acf'](self.prefix.diff(12).iloc[-48:].to_numpy(),nlags=12,fft=False,adjusted=False,missing='raise')
        self.assertEqual(actual['acf_lag12'],float(expected[12]))
        r = h['residual_summary']({'dates':self.prefix.index[-48:].strftime('%Y-%m-%d').tolist(),'values':[1.]*48},self.prefix.index[-48:])
        self.assertIsNone(r['acf_lag1'])
        with self.assertRaises(ValueError):
            h['residual_summary']({'dates':self.prefix.index[-48:].strftime('%Y-%m-%d').tolist(),'values':[1.]*47},self.prefix.index[-48:])
    def test_information_boundary_mutation(self):
        panel = pd.read_csv(h['CFG'].data,parse_dates=['date'])
        altered = panel.copy(deep=True)
        mask = altered.date > '2024-06-01'
        altered.loc[mask,'turnover_aud_millions'] *= 1000
        self.assertFalse(panel.loc[mask,'turnover_aud_millions'].equals(altered.loc[mask,'turnover_aud_millions']))
        sid = h['CASE_SPECS'][0]['series_id']
        a,b = [h['prefix_from_panel'](p,sid) for p in (panel,altered)]
        self.assertTrue(a.equals(b))
        outputs=[]
        for prefix in (a,b):
            s = h['DiagnosticSession'](prefix,self.meta,fitter=h['fixture_fit'])
            outputs.append(h['run_adaptive'](s,self.protocol,client=h['scripted_client']([h['scripted_action']('compare_recent_history'),h['scripted_action']('stop',ids=['o1'])])))
        self.assertEqual(h['semantic_record'](outputs[0]),h['semantic_record'](outputs[1]))
        changed=h['DiagnosticSession'](a*1.01,self.meta,fitter=h['fixture_fit'])
        with self.assertRaises(h['ReplayMismatch']):
            h['run_adaptive'](changed,self.protocol,recorded=outputs[0])
        with self.assertRaises(h['ReplayMismatch']):
            h['run_adaptive'](h['DiagnosticSession'](a,self.meta,fitter=h['fixture_fit']),{'changed_schema': True},recorded=outputs[0])
    def test_failure_fixture_boundaries(self):
        rows, r=h['failure_fixtures'](self.protocol)
        self.assertEqual(len(rows),14)
        for name in ('Malformed JSON','Unknown action','Unknown evidence','Prerequisite','Request timeout','Sixth tool rejected'):
            self.assertEqual(r[name]['terminal']['status'],'failed',name)
        self.assertEqual(r['Duplicate call']['work']['tool_attempts'],2)
        self.assertEqual(r['Duplicate call']['work']['duplicate_tools'],1)
        self.assertEqual(r['Tool exception']['work']['failed_tools'],1)
        self.assertEqual(r['Five tools then stop']['work']['model_requests'],6)
        self.assertEqual(r['Five tools then stop']['work']['tool_attempts'],5)
        self.assertEqual(r['Five tools then stop']['terminal']['status'],'completed')
        self.assertEqual(r['Sixth tool rejected']['work']['tool_attempts'],5)
        self.assertEqual(r['Admission expiry']['work'].get('model_requests',0),0)
        self.assertEqual(r['Final-fit fallback']['forecast_status'],'fallback')
    def test_replay_preserves_timeout_and_fixed_prefix(self):
        r=self.run_actions([TimeoutError('fixture timeout')])
        replay=h['run_adaptive'](self.session(),self.protocol,recorded=r,client=lambda *a: self.fail('No live call in replay'))
        self.assertEqual(h['semantic_record'](replay),h['semantic_record'](r))
        b=h['run_checklist'](self.session())
        replay=h['run_checklist'](self.session(),recorded=b)
        self.assertEqual(h['semantic_record'](b),h['semantic_record'](replay))
        partial=copy.deepcopy(b)
        partial['steps']=[]
        s=self.session()
        partial['tested_candidates']=s.tested
        partial['delivery']=s.finalize()
        partial['terminal']={'status':'budget_exhausted','stop_source':'budget','reason':'admission_before_tool'}
        partial['semantic_hash']=h['digest'](h['semantic_record'](partial))
        replay=h['run_checklist'](self.session(),recorded=partial)
        self.assertEqual(len(replay['steps']),0)
        self.assertEqual(replay['terminal']['status'],'budget_exhausted')
    def test_reference_completeness(self):
        b=h['run_checklist'](self.session())
        c=self.run_actions([h['scripted_action']('stop')])
        comparison=h['selection_comparison'](b,c)
        self.assertTrue(comparison['full_menu_complete'])
        self.assertGreaterEqual(comparison['regret'],0)
        b['tested_candidates'].pop('hw_full')
        result=h['selection_comparison'](b,c)
        self.assertIsNone(result['regret'])
        self.assertEqual(result['reason'],'Fixed menu incomplete')
    def test_timeout_clipped_and_before_tool(self):
        class Clock:
            value=0
            def __call__(self): return self.value
        clock=Clock()
        calls=[]
        class SetupClock:
            n=0
            def __call__(self):
                self.n+=1
                return 0 if self.n==1 else 290
        h['run_adaptive'](self.session(),self.protocol,
            client=lambda m,t: calls.append(t) or {'text':h['json_text'](h['scripted_action']('stop'))},clock=SetupClock())
        self.assertEqual(calls,[10])
        def slow(m,t):
            clock.value=301
            return {'text':h['json_text'](h['scripted_action']('compare_recent_history'))}
        r=h['run_adaptive'](self.session(),self.protocol,client=slow,clock=clock)
        self.assertEqual(r['terminal']['reason'],'admission_before_tool')
        self.assertEqual(r['work'].get('tool_attempts',0),0)
    def test_numerical_overshoot_and_preserved_budget(self):
        class Clock:
            value=0
            def __call__(self): return self.value
        clock=Clock()
        session=self.session()
        normal=session.compare_recent_history
        def slow_tool():
            result=normal()
            clock.value=301
            return result
        session.compare_recent_history=slow_tool
        record=h['run_adaptive'](session,self.protocol,client=h['scripted_client']([h['scripted_action']('compare_recent_history')]),clock=clock)
        self.assertEqual(record['work']['tool_attempts'],1)
        self.assertEqual(record['terminal']['status'],'budget_exhausted')
        self.assertEqual(record['timing']['overshoot_seconds'],1)
        replay=h['run_adaptive'](self.session(),self.protocol,recorded=record)
        self.assertEqual(h['semantic_record'](record),h['semantic_record'](replay))
        clock.value=0
        session=self.session()
        normal=session.decompose_history
        def slow_stl():
            result=normal()
            clock.value=301
            return result
        session.decompose_history=slow_stl
        fixed=h['run_checklist'](session,clock=clock)
        self.assertEqual(fixed['work']['tool_attempts'],1)
        self.assertEqual(fixed['terminal']['status'],'budget_exhausted')
        self.assertFalse(h['selection_comparison'](fixed,record)['full_menu_complete'])
    def test_duplicate_json_fields_and_invalid_prefix(self):
        with self.assertRaises(ValueError): h['strict_json']('{"action":1,"action":2}')
        with self.assertRaises(ValueError): h['strict_json']('{"value":NaN}')
        with self.assertRaises(ValueError): h['DiagnosticSession'](self.prefix.iloc[:-1],self.meta)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    h['save_json'](h['CFG'].evidence/'development_checks.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'core_hash':h['core_identity'](),
        'environment':h['environment_record'](), 'source':'scripts/module03/check_development.py'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
