"""Synthetic boundary tests; no private corpus or actual embedding model required."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('companion_worker', ROOT / 'scripts/companion_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class CompanionWorkerTests(unittest.TestCase):
    def row(self, **patch):
        row = {
            'id': 'r1', 'window_id': 'w1', 'source_id': 's1', 'speaker_id': 'nailv',
            'author': 'streamer', 'kind': 'quote', 'layer': 3,
            'review_status': 'text_reviewed', 'epistemic_status': 'expression',
            'speaker_verified': False,
            'evidence': [{'window_id': 'w1', 'start_char': 0, 'end_char': 5,
                          'text': 'hello', 'author': 'streamer'}],
            'trigger_evidence': [], 'text': 'greeting'
        }
        row.update(patch)
        return row

    def test_protocol_limits_and_scope(self):
        self.assertEqual('search', worker.validate_payload({'id': 'x', 'action': 'search', 'query': 'hello'})['action'])
        for value in [{'id': 'x', 'query': ''}, {'id': 'x', 'query': 'hi', 'scope': 'all'},
                      {'id': 'x', 'query': 'hi', 'top_k': True}, {'id': 'x', 'action': 'repair'}]:
            with self.assertRaises(ValueError):
                worker.validate_payload(value)

    def test_exact_evidence_and_non_memory_projection(self):
        out = worker.safe_candidate(self.row(), {'w1': {'raw_text': 'hello world'}})
        self.assertEqual('hello', out['evidence'][0]['text'])
        self.assertFalse(out['can_become_character_memory'])
        self.assertNotIn('raw_text', out)
        self.assertFalse(out['speaker_verified'])

    def test_hearsay_hypothetical_context_only_excluded(self):
        for patch in [{'epistemic_status': 'hearsay'}, {'epistemic_status': 'hypothetical'},
                      {'review_status': 'context_only'}, {'evidence': []}]:
            self.assertIsNone(worker.safe_candidate(self.row(**patch), {'w1': {'raw_text': 'hello'}}))

    def test_tampered_or_misattributed_evidence_fails_closed(self):
        with self.assertRaises(ValueError):
            worker.safe_candidate(self.row(), {'w1': {'raw_text': 'wrong'}})
        row = self.row()
        row['evidence'][0]['author'] = 'viewer'
        self.assertIsNone(worker.safe_candidate(row, {'w1': {'raw_text': 'hello'}}))

    def test_budget_selects_complete_spans_not_cut_quotes(self):
        self.assertIsNone(worker.safe_candidate(self.row(), {'w1': {'raw_text': 'hello'}}, max_chars=4))


if __name__ == '__main__':
    unittest.main()
