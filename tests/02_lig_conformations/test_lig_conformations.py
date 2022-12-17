import unittest
import os
from ase import Atom
from architector.io_lig import get_aligned_conformer


coord_list = [
            [2., 0.0, 0.0],
            [0.0, 2., 0.0],
            [0.0, 0.0, 2.],
            [-2., 0.0, 0.0],
            [0.0, -2., 0.0],
            [0.0, 0.0, -2.]
        ]

############### First Test!
# Octahedral
# 0,3 should be opposite
# 1,4 should be opposite
# 2,5 should be opposite
class Test03LigConfirmations1(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.remove('FeBipy3.xyz')

    def test_input(self):
        # It would be nice to separate the different tests in different functions but currently all results are merged
        # at the end so a separation is not possible.
        test_input = {
            "smiles": "n1ccccc1-c2ccccn2",
            "ligcoordList": [[0, 0], [11, 1]],
            "corecoordList": coord_list,
            "metal": 'Fe'
        }

        test_input1 = {
            "smiles": "n1ccccc1-c2ccccn2",
            "ligcoordList": [[0, 2], [11, 3]],
            "corecoordList": coord_list,
            "metal": 'Fe'
        }

        test_input2 = {
            "smiles":"n1ccccc1-c2ccccn2",
            "ligcoordList": [[0, 4], [11, 5]],
            "corecoordList": coord_list,
            "metal": 'Fe'
        }

        test_inputs = [test_input, test_input1, test_input2]

        out = []
        for test in test_inputs:
            outatoms, minval, sane, final_relax, bo_dict, atypes, _ = get_aligned_conformer(
                test['smiles'],
                test['ligcoordList'],
                test['corecoordList'],
                metal=test['metal'],
            )
            out.append(outatoms)

        tmpats = out[0].copy()
        for i in out[1:]:
            tmpats += i

        m_at = Atom(test_input['metal'], (0, 0, 0))
        tmpats.append(m_at)

        tmpats.write('FeBipy3.xyz')


# ################ Second Test -> Shift to P, U
# class Test03LigConfirmations2(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.metal = 'U'

#     @classmethod
#     def tearDownClass(cls):
#         os.remove('UBiPy3.xyz')

#     def test_input(self):
#         test_input = {
#             "smiles": "n1ccccc1-c2ccccp2",
#             "ligcoordList": [[0, 0], [11, 1]],
#             "corecoordList": coord_list,
#             "metal": self.metal
#         }

#         test_input1 = {
#             "smiles": "n1ccccc1-c2ccccp2",
#             "ligcoordList": [[0, 2], [11, 3]],
#             "corecoordList": coord_list,
#             "metal": self.metal
#         }

#         test_input2 = {
#             "smiles": "n1ccccc1-c2ccccp2",
#             "ligcoordList": [[0, 4], [11, 5]],
#             "corecoordList": coord_list,
#             "metal": self.metal
#         }

#         test_inputs = [test_input, test_input1, test_input2]

#         out = []
#         for test in test_inputs:
#             outatoms, minval, sane, final_relax, bo_dict, atypes, _ = get_aligned_conformer(
#                 test['smiles'],
#                 test['ligcoordList'],
#                 test['corecoordList'],
#                 metal=test['metal'],
#             )
#             out.append(outatoms)

#         tmpats = out[0].copy()
#         for i in out[1:]:
#             tmpats += i

#         m_at = Atom(test_input['metal'], (0,0,0))
#         tmpats.append(m_at)

#         tmpats.write('UBiPy3.xyz')


################ Third Test - > Flip one BiPY
class Test03LigConfirmations3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metal = 'U'

    @classmethod
    def tearDownClass(cls):
        os.remove('UBiPy3_flip_one.xyz')

    def test_input(self):
        test_input = {
            "smiles": "n1ccccc1-c2ccccp2",
            "ligcoordList": [[0, 1], [11, 0]],
            "corecoordList": coord_list,
            "metal": self.metal}  # Flip One

        test_input1 = {
            "smiles": "n1ccccc1-c2ccccp2",
            "ligcoordList": [[0, 2], [11, 3]],
            "corecoordList": coord_list,
            "metal": self.metal
        }

        test_input2 = {
            "smiles": "n1ccccc1-c2ccccp2",
            "ligcoordList": [[0, 4], [11, 5]],
            "corecoordList": coord_list,
            "metal": self.metal
        }

        test_inputs = [test_input, test_input1, test_input2]

        out = []
        for test in test_inputs:
            outatoms, minval, sane, final_relax, bo_dict, atypes, _ = get_aligned_conformer(
                test['smiles'],
                test['ligcoordList'],
                test['corecoordList'],
                metal=test['metal'],
            )
            out.append(outatoms)

        tmpats = out[0].copy()
        for i in out[1:]:
            tmpats += i

        m_at = Atom(test_input['metal'], (0, 0, 0))
        tmpats.append(m_at)

        tmpats.write('UBiPy3_flip_one.xyz')


################ Fourth Test - > Am, with Tetradentate!
class Test03LigConfirmations4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metal = 'Am'

    @classmethod
    def tearDownClass(cls):
        os.remove('Am4_2.xyz')

    def test_input(self):
        test_input = {
            "smiles": "n1ccccc1-c2ccccp2",
            "ligcoordList": [[0, 1], [11, 0]],
            "corecoordList": coord_list,
            "metal": self.metal
        }  # Flip One

        test_input1 = {
            "smiles": 'C(CO)N(CCO)CCO',  # Triethanolamine
            "ligcoordList": [
                [3, 4],
                [9, 2],
                [6, 5],
                [2, 3]
            ],
            "corecoordList": coord_list,
            "metal": self.metal
        }

        test_inputs = [test_input, test_input1]

        out = []
        for test in test_inputs:
            outatoms, minval, sane, final_relax, bo_dict, atypes, _ = get_aligned_conformer(
                test['smiles'],
                test['ligcoordList'],
                test['corecoordList'],
                metal=test['metal'],
            )
            out.append(outatoms)

        tmpats = out[0].copy()
        for i in out[1:]:
            tmpats += i

        m_at = Atom(test_input['metal'], (0, 0, 0))
        tmpats.append(m_at)

        tmpats.write('Am4_2.xyz')


# ################ Fifth Test - > Fe, with Tetradentate!
# class Test03LigConfirmations5(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.metal = 'Fe'

#     @classmethod
#     def tearDownClass(cls):
#         os.remove('Fe4_2.xyz')

#     def test_input(self):
#         test_input = {
#             "smiles": "n1ccccc1-c2ccccp2",
#             "ligcoordList": [[0, 0], [11, 1]],
#             "corecoordList": coord_list,
#             "metal": self.metal
#         }  # Flip One

#         test_input1 = {
#             "smiles": 'C(CO)N(CCO)CCO',  # Triethanolamine
#             "ligcoordList": [
#                 [3, 4],
#                 [9, 2],
#                 [6, 5],
#                 [2, 3]
#             ],
#             "corecoordList": coord_list,
#             "metal": self.metal
#         }

#         test_inputs = [test_input, test_input1]

#         out = []
#         for test in test_inputs:
#             outatoms, minval, sane, final_relax, bo_dict, atypes, _ = get_aligned_conformer(
#                 test['smiles'],
#                 test['ligcoordList'],
#                 test['corecoordList'],
#                 metal=test['metal'],
#             )
#             out.append(outatoms)

#         tmpats = out[0].copy()
#         for i in out[1:]:
#             tmpats += i

#         m_at = Atom(test_input['metal'], (0, 0, 0))
#         tmpats.append(m_at)

#         tmpats.write('Fe4_2.xyz')


################ Sixth Test - > Mo, with Hexadentate!
class Test03LigConfirmations6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metal = 'Mo'

    @classmethod
    def tearDownClass(cls):
        os.remove('Mo_hexadentate.xyz')

    def test_input(self):
        test_input = {
            "smiles": 'C(CN(CC(=O)O)CC(=O)O)N(CCN(CC(=O)O)CC(=O)O)CC(=O)O',  # DTPA
            "ligcoordList": [
                [26, 0],
                [10, 5],
                [22, 1],
                [18, 2],
                [6, 3],
                [2, 4]
            ],
            "corecoordList": coord_list,
            "metal": self.metal
        }  # Flip One

        test_inputs = [test_input]

        out = []
        for test in test_inputs:
            outatoms, minval, sane, final_relax, bo_dict, atypes, _ = get_aligned_conformer(
                test['smiles'],
                test['ligcoordList'],
                test['corecoordList'],
                metal=test['metal'],
            )
            out.append(outatoms)

        tmpats = out[0].copy()
        for i in out[1:]:
            tmpats += i

        m_at = Atom(test_input['metal'], (0, 0, 0))
        tmpats.append(m_at)

        tmpats.write('Mo_hexadentate.xyz')


if __name__ == "__main__":
    unittest.main()
