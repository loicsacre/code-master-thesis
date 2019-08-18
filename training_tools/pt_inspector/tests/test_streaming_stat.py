import numpy as np

from pt_inspector import StreamingStat

import unittest


class StreamingStatTester(unittest.TestCase):

    def test_before_running(self):
        # np.random.seed(0)
        x = np.random.uniform(size=(1, 10))
        steaming_stat = StreamingStat()
        steaming_stat.add(x[0])

        # Overall
        mean, std = steaming_stat.get_running()
        self.assertAlmostEqual(x.mean(), mean)
        self.assertAlmostEqual(x.std(), std)

    def test_running(self):
        running_state = StreamingStat()
        for _ in range(1):  # Twice to test running_state.reset()
            X = np.random.uniform(size=(5, 10))
            for x in X:
                # Adding into running stat
                running_state.add(x)

            # First
            mean, std = running_state.get_first()
            self.assertAlmostEqual(X[0].mean(), mean)
            self.assertAlmostEqual(X[0].std(), std)

            # Last
            mean, std = running_state.get_running()
            self.assertAlmostEqual(X.mean(), mean)
            self.assertAlmostEqual(X.std(), std)

            # Overall
            mean, std = running_state.get_last()
            self.assertAlmostEqual(X[-1].mean(), mean)
            self.assertAlmostEqual(X[-1].std(), std)

            running_state.reset()


if __name__ == '__main__':
    unittest.main()
