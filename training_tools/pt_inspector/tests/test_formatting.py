from pt_inspector.formatting import format_tree_view

import unittest


class FormatTreeViewTester(unittest.TestCase):
    def generator(self):
        yield 'features.Conv.weight', 0
        yield 'features.BN.weight', 1
        yield 'features.BN.bias', 3
        yield 'features.denseblock_1.denselayer_1.BT-BN.weight', 4
        yield 'features.denseblock_1.denselayer_1.BT-Conv.weight', 5
        yield 'features.denseblock_1.denselayer_2.BT-BN.weight', 6
        yield 'features.denseblock_2.denselayer_1.BT-BN.weight', 7
        yield 'features.Norm.weight', 8

    def expected_result(self):
        yield 'features', None
        yield '|- Conv', None
        yield '|  |- weight', 0
        yield '|- BN', None
        yield '|  |- weight', 1
        yield '|  |- bias', 3
        yield '|- denseblock_1', None
        yield '|  |- denselayer_1', None
        yield '|  |  |- BT-BN', None
        yield '|  |  |  |- weight', 4
        yield '|  |  |- BT-Conv', None
        yield '|  |  |  |- weight', 5
        yield '|  |- denselayer_2', None
        yield '|  |  |- BT-BN', None
        yield '|  |  |  |- weight', 6
        yield '|- denseblock_2', None
        yield '|  |- denselayer_1', None
        yield '|  |  |- BT-BN', None
        yield '|  |  |  |- weight', 7
        yield '|- Norm', None
        yield '|  |- weight', 8

    def test_view(self):
        for given, expected in zip(format_tree_view(self.generator()),
                                   self.expected_result()):
            #print('{:<50}{:<50}'.format(expected[0], given[0]))
            self.assertEqual(given, expected)


if __name__ == '__main__':
    unittest.main()
