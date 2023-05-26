"""Testing assignment generation"""

import unittest

import numpy as np

from uat.assignment import SimpleNewAssGenerator


class TestAssignmentGenerators(unittest.TestCase):
    """Test contour functionality"""

    def test_new(self):
        models = []

        ass_gen = SimpleNewAssGenerator(models)

        targets = np.arange(0, 10, 1, dtype=np.int32)

        # test that empty model leads to problems
        with self.assertRaises(ValueError):
            ass_gen.generate(None, None, targets)

        def simple_score(tr, s, t):
            # pylint: disable=unused-argument
            return np.zeros((len(t),), dtype=np.float32)

        models = [simple_score]

        ass_gen = SimpleNewAssGenerator(models)

        ass_gen.generate(None, None, targets)


if __name__ == "__main__":
    unittest.main()
