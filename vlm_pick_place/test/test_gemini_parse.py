import unittest

from vlm_pick_place.gemini_brain import extract_json, format_history, parse_step_decision


class TestGeminiParse(unittest.TestCase):
    def test_extract_from_code_fence(self):
        text = 'Sure!\n```json\n{"done": true, "reason": "ok"}\n```\n'
        self.assertEqual(extract_json(text), {'done': True, 'reason': 'ok'})

    def test_extract_list(self):
        self.assertEqual(extract_json('[{"point": [1, 2], "label": "a"}]'),
                         [{'point': [1, 2], 'label': 'a'}])

    def test_parse_step(self):
        d = parse_step_decision({
            'done': False,
            'pick': {'label': 'red cube', 'point': [400, 520]},
            'place': {'label': 'red tray', 'point': [300, 700]},
            'reason': 'red goes to red', 'say_ja': '赤を赤へ'})
        self.assertFalse(d.done)
        self.assertEqual(d.pick.label, 'red cube')
        self.assertEqual(d.pick.point_yx, (400.0, 520.0))
        self.assertEqual(d.place.point_yx, (300.0, 700.0))

    def test_parse_done_without_targets(self):
        d = parse_step_decision([{'done': True, 'reason': 'all sorted'}])
        self.assertTrue(d.done)
        self.assertIsNone(d.pick)

    def test_rejects_bad_points(self):
        for bad in ([1], [1, 2, 3], ['a', 'b'], [-5, 10], [100, 1200]):
            with self.assertRaises(ValueError):
                parse_step_decision({'done': False,
                                     'pick': {'label': 'x', 'point': bad},
                                     'place': {'label': 'y', 'point': [1, 1]}})

    def test_rejects_missing_place(self):
        with self.assertRaises(ValueError):
            parse_step_decision({'done': False, 'pick': {'label': 'x', 'point': [1, 1]}})

    def test_format_history(self):
        self.assertEqual(format_history([]), '(none yet)')
        self.assertEqual(format_history(['a', 'b']), '1. a\n2. b')


if __name__ == '__main__':
    unittest.main()
