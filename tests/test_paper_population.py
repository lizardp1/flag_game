import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.run_paper_population import plans
from nnd.flag_game.backend import FlagGameOpenAIBackend


class PaperPopulationTests(unittest.TestCase):
    def test_design_and_resolution(self):
        for baseline, counts in [('gpt4o', [38,38,38,38,38,37]),
                                 ('gpt54', [29,29,29,29,29,26])]:
            configs = plans('gpt-5.6-terra', baseline, Path('runs/test'), reasoning_effort='none')
            self.assertEqual([len(c.seeds) for c in configs], counts)
            self.assertEqual([c.N for c in configs], [4,8,16,32,64,128])
            self.assertEqual([c.rounds for c in configs], [20,20,24,32,32,32])
            for c in configs:
                self.assertEqual(len(c.seeds), len(set(c.seeds)))
                self.assertFalse(set(c.seeds) & {12,16,17,19,25,29})
                resolved = c.resolve(c.seeds[0])
                self.assertEqual(resolved.probe_every, c.N//2)
                self.assertEqual(resolved.T, c.rounds*c.N)
                self.assertEqual(resolved.reasoning_effort, 'none')
                self.assertEqual(resolved.temperature, .2)
                self.assertEqual(resolved.early_stop_probe_window, 5)
                self.assertEqual(resolved.country_pool, 'stripe_expanded_24')
                self.assertEqual(resolved.agent_models, ['gpt-5.6-terra']*c.N)

    def test_smoke_is_bounded(self):
        configs = plans('gpt-5.6-terra', 'gpt4o', Path('runs/smoke'), smoke=True)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].seeds, [1])
        self.assertEqual(configs[0].resolve(1).T, 4)
        self.assertEqual(configs[0].early_stop_window, 0)

    def test_sampling_request_and_optional_reasoning(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = object.__new__(FlagGameOpenAIBackend)
            backend.model = 'gpt-5.6-terra'
            backend.temperature = .2
            backend.top_p = 1.
            backend.max_tokens = 200
            backend.debug_dir = Path(tmp)
            backend._audit_lock = threading.Lock()
            backend._record_usage = Mock()
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))], model='provider')
            create = Mock(return_value=response)
            backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            for effort in [None, 'none']:
                backend.reasoning_effort = effort
                self.assertEqual(backend._call([]), '{}')
                args = create.call_args.kwargs
                self.assertEqual(args['temperature'], .2)
                self.assertEqual(args['top_p'], 1.)
                self.assertEqual(args['max_completion_tokens'], 200)
                if effort is None:
                    self.assertNotIn('reasoning_effort', args)
                else:
                    self.assertEqual(args['reasoning_effort'], 'none')
