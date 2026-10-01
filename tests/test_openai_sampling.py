import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from nnd.flag_game.backend import FlagGameOpenAIBackend


class OpenAISamplingTests(unittest.TestCase):
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
