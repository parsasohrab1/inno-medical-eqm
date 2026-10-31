import unittest
import numpy as np
from inno_medical_eqm import core


class TestEqm(unittest.TestCase):
    def test_plan_respects_capacity_and_orders_by_priority(self):
        devs = [{"id": "a", "type": "ventilator", "p": 0.9}, {"id": "b", "type": "ecg", "p": 0.8}, {"id": "c", "type": "autoclave", "p": 0.1}]
        pl = core.plan(devs, 4)
        self.assertLessEqual(sum(x["hours"] for x in pl), 4)
        self.assertEqual(pl[0]["id"], "b")  # best risk*criticality per hour

    def test_plan_zero_capacity(self):
        self.assertEqual(core.plan([{"id": "a", "type": "ventilator", "p": 1.0}], 0), [])

    def test_health_index_bounds(self):
        h = core.health_index([0, 0.25, 1])
        self.assertTrue(((h >= 0) & (h <= 100)).all())

    def test_model_monotone_in_vibration(self):
        t, X, y = core.simulate(1)
        m = core.fit(X, y)
        a = X[:1].copy(); b = a.copy(); b[0, 2] += 2.0
        self.assertGreater(core.predict(m, b)[0], core.predict(m, a)[0])

    def test_expected_failures_sum(self):
        d = [{"id": 1, "type": "ecg", "p": .2}, {"id": 2, "type": "ecg", "p": .3}]
        self.assertAlmostEqual(core.expected_failures_by_type(d)["ecg"], 0.5)
