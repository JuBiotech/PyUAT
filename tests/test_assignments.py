"""Testing assignment generation"""

import unittest

import numpy as np

from uatrack.assignment import SimpleNewAssGenerator, filter_targets


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

        assignments = ass_gen.generate(None, None, targets)

        self.assertEqual(len(assignments[0]), 10)

    def test_filter(self):

        # dummy example to filter out some assignments
        sources = np.arange(0, 10, 1, dtype=np.uint32)
        targets = np.copy(sources)

        source_index = sources
        target_index = target_index = np.tile(np.array(targets), (len(sources), 1))

        # pylint: disable=unused-argument
        def filter(a, b):
            return b < 5

        filters = [filter]

        new_source_index, new_target_index = filter_targets(
            source_index, target_index, filters=filters
        )

        self.assertEqual(len(new_source_index), 10)
        self.assertEqual(new_target_index.size, 50)


if __name__ == "__main__":
    unittest.main()
