import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nnd.experiment_cli import Experiment, execute, prompt_examples
from nnd.flag_game_org import prompts, parsing
from nnd.flag_game_org.backend import OrgFlagGameOpenAIBackend
from nnd.flag_game_broadcast.parsing import parse_broadcast_statement
from nnd.backends.parsing import ParseError


class ExperimentTests(unittest.TestCase):
    def config(self, protocol='manager', **kwargs):
        return Experiment(protocol=protocol,N=2,composition={'gpt-4o':1,'gpt-5.4-2026-03-05':1},
                          country_pool='stripe_expanded_24',render_scale=1,rounds=2,save_crop_images=False,**kwargs)

    def test_validation(self):
        with self.assertRaises(ValueError): Experiment(protocol='manager',N=2,composition={'gpt-4o':1})
        with self.assertRaises(ValueError): self.config(social_evidence_alpha=1.1)
        with self.assertRaises(ValueError): self.config(message_bandwidth=4)
        with self.assertRaises(ValueError): self.config(typo=True)
        with self.assertRaises(ValueError): self.config(backend='anthropic')

    def test_manager_roles_and_allocation(self):
        cfg=self.config().resolve(3)
        self.assertEqual(len(cfg.agent_models),3)
        self.assertEqual(cfg.agent_models[0],'gpt-4o')
        self.assertEqual(cfg.agent_models,self.config().resolve(3).agent_models)

    def test_manager_bandwidth_wire_format(self):
        for m in [1,2,3]:
            obj={'country':'France'}
            if m>1:obj['reason']='vertical stripes'
            for parser in [parsing.parse_observer_statement,parsing.parse_organization_decision]:
                parsed=parser(json.dumps(obj),countries=['France','Peru'],m=m)
                self.assertEqual(parsed.normalized_memory_entry(),'France' if m==1 else 'France | vertical stripes')
                wrong={**obj,'extra':'unexpected'}
                with self.assertRaises(ParseError):parser(json.dumps(wrong),countries=['France'],m=m)
        with self.assertRaises(ParseError):parsing.parse_observer_statement('{"country":"France","reason":"leak"}',countries=['France'],m=1)

    def test_guidance_off_and_on(self):
        off=prompt_examples(self.config(social_guidance_enabled=False))
        low=prompt_examples(self.config(social_guidance_enabled=True,social_evidence_alpha=0))
        high=prompt_examples(self.config(social_guidance_enabled=True,social_evidence_alpha=1))
        self.assertNotIn('weak evidence',off['observer'])
        self.assertIn("manager's past decisions as weak evidence",low['observer'])
        self.assertIn('current observer reports as strong evidence',high['manager'])
        self.assertNotIn('your own crop',high['manager'])
        self.assertNotEqual(low['manager'],high['manager'])

    def test_real_backend_forwards_guidance_and_keeps_manager_blind(self):
        # Exercise real message construction without constructing a client or making a call.
        backend=object.__new__(OrgFlagGameOpenAIBackend)
        backend.social_susceptibility=1
        backend.prompt_social_susceptibility=True
        backend.image_detail='high'
        seen=[]
        def capture(messages,parser,**kw):
            seen.append(messages)
            return parser('{"country":"France"}')
        backend._call_with_retries=capture
        backend.observer_statement(countries=['France'],prepared_crop='data:image/png;base64,test',memory_lines=[],m=1)
        backend.organization_decision(countries=['France'],memory_lines=[],observer_statement_lines=['{"country":"France"}'],m=1)
        self.assertIn('strong evidence',seen[0][1]['content'][0]['text'])
        self.assertIsInstance(seen[1][1]['content'],str)
        self.assertNotIn('image_url',json.dumps(seen[1]))

    def test_anonymous_broadcast_contract(self):
        rendered=json.dumps(prompt_examples(self.config('broadcast',social_guidance_enabled=True)))
        for forbidden in ['model_identity','gpt-','SOTA','Social susceptibility a =']:
            self.assertNotIn(forbidden,rendered)
        for m in [1,2,3]:
            raw={'country':'France'}
            if m>1:raw['reason']='red white blue'
            parsed=parse_broadcast_statement(json.dumps(raw),countries=['France'],m=m)
            self.assertNotIn('gpt',parsed.normalized_broadcast())

    def test_all_protocol_bandwidths_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            for protocol in ['pairwise','broadcast','manager']:
                for m in [1,2,3]:
                    cfg=self.config(protocol,message_bandwidth=m,social_guidance_enabled=True,
                                    output_root=Path(tmp)/f'{protocol}_{m}')
                    execute(cfg)
                    out=cfg.output_root/'seed_0000'
                    record=json.loads((out/'experiment.json').read_text())
                    self.assertEqual(record['status'],'complete')
                    self.assertEqual(record['summary']['api_call_count'],0)
                    self.assertTrue((out/'trial_manifest.json').exists())
                    before=(out/'experiment.json').read_bytes()
                    with self.assertRaises(ValueError):execute(cfg)
                    execute(cfg,resume=True)
                    self.assertEqual(before,(out/'experiment.json').read_bytes())
                    if protocol=='manager':
                        rows=[json.loads(x) for x in (out/'observations.jsonl').read_text().splitlines()]
                        self.assertTrue(all(r['valid'] for r in rows))
                        self.assertTrue(all(bool(r['reason'])==(m>1) for r in rows))


class PublicCliTests(unittest.TestCase):
    def test_cli_sweeps_are_offline(self):
        from typer.testing import CliRunner
        from nnd.experiment_cli import app
        runner=CliRunner()
        root=Path(__file__).resolve().parents[1]
        for filename,count in [('alpha_composition.yaml',45),('population.yaml',12),('bandwidth.yaml',9)]:
            result=runner.invoke(app,['sweep','--config',str(root/'experiments/social'/filename),'--dry-run'])
            self.assertEqual(result.exit_code,0,result.output)
            self.assertEqual(len(json.loads(result.output)),count)

    def test_mixed_broadcast_cannot_expose_fixed_model_slots(self):
        with self.assertRaises(ValueError):
            Experiment(protocol='broadcast',N=2,composition={'gpt-4o':1,'gpt-5.4-2026-03-05':1},randomize_model_slots=False)
        with self.assertRaises(ValueError):
            Experiment(protocol='manager',N=2,composition={'gpt-4o':2.0})

    def test_provider_returned_model_is_recorded(self):
        from types import SimpleNamespace
        from nnd.flag_game.backend import FlagGameOpenAIBackend
        backend=object.__new__(FlagGameOpenAIBackend)
        backend.model='gpt-4o';backend.usage_rows=[]
        backend._record_usage(SimpleNamespace(model='gpt-4o-test-snapshot',usage=SimpleNamespace(prompt_tokens=1,completion_tokens=2,total_tokens=3)))
        self.assertEqual(backend.usage_summary()['provider_model_ids'],['gpt-4o-test-snapshot'])

    def test_legacy_manager_prompt_unchanged_when_unprompted(self):
        text=prompts.observer_statement_text(countries=['France'],memory_lines=[],m=3)
        self.assertNotIn('weak evidence',text)
        self.assertTrue(text.endswith('{"country":"<one allowed country>","reason":"<one sentence describing what you see>"}'))
        self.assertFalse(ExperimentTests().config().resolve(0).prompt_social_susceptibility)

if __name__=='__main__':unittest.main()
