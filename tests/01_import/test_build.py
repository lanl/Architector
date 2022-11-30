
"""
Try building a complex through importing architector.
"""

# Imports
import unittest
import architector.complex_construction as arch_bc
import architector.io_align_mol as io_align_mol
import shutil
import os

refmol = """
@<TRIPOS>MOLECULE
user_core
    61    69     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Fe1       0.0000    0.0000    0.0000   Fe        1 RES1   0.0000
     2 N1        0.0000   -1.9635   -0.0578   N.ar      1 RES1   0.0000
     3 C1        0.0000   -2.8705    0.9419   C.ar      1 RES1   0.0000
     4 C2       -0.0000   -4.2358    0.7384   C.ar      1 RES1   0.0000
     5 C3       -0.0000   -4.6943   -0.5697   C.ar      1 RES1   0.0000
     6 C4       -0.0000   -3.7747   -1.6208   C.ar      1 RES1   0.0000
     7 C5       -0.0000   -2.4031   -1.3474   C.ar      1 RES1   0.0000
     8 C6       -0.0000   -1.3474   -2.4031   C.ar      1 RES1   0.0000
     9 C7       -0.0000   -1.6209   -3.7747   C.ar      1 RES1   0.0000
    10 C8       -0.0000   -0.5697   -4.6943   C.ar      1 RES1   0.0000
    11 C9        0.0000    0.7384   -4.2358   C.ar      1 RES1   0.0000
    12 C10       0.0000    0.9419   -2.8705   C.ar      1 RES1   0.0000
    13 N2       -0.0000   -0.0578   -1.9635   N.ar      1 RES1   0.0000
    14 H1        0.0000   -2.4500    1.9432   H         1 RES1   0.0000
    15 H2       -0.0000   -4.9221    1.5771   H         1 RES1   0.0000
    16 H3       -0.0000   -5.7609   -0.7762   H         1 RES1   0.0000
    17 H4       -0.0000   -4.1563   -2.6344   H         1 RES1   0.0000
    18 H5       -0.0000   -2.6344   -4.1563   H         1 RES1   0.0000
    19 H6       -0.0000   -0.7762   -5.7609   H         1 RES1   0.0000
    20 H7        0.0000    1.5771   -4.9221   H         1 RES1   0.0000
    21 H8        0.0000    1.9432   -2.4500   H         1 RES1   0.0000
    22 N3       -0.0578    0.0000    1.9635   N.ar      1 RES1   0.0000
    23 C11       0.9419   -0.0000    2.8705   C.ar      1 RES1   0.0000
    24 C12       0.7384    0.0000    4.2358   C.ar      1 RES1   0.0000
    25 C13      -0.5697    0.0000    4.6943   C.ar      1 RES1   0.0000
    26 C14      -1.6208    0.0000    3.7747   C.ar      1 RES1   0.0000
    27 C15      -1.3474    0.0000    2.4031   C.ar      1 RES1   0.0000
    28 C16      -2.4031    0.0000    1.3474   C.ar      1 RES1   0.0000
    29 C17      -3.7747    0.0000    1.6209   C.ar      1 RES1   0.0000
    30 C18      -4.6943    0.0000    0.5697   C.ar      1 RES1   0.0000
    31 C19      -4.2358   -0.0000   -0.7384   C.ar      1 RES1   0.0000
    32 C20      -2.8705   -0.0000   -0.9419   C.ar      1 RES1   0.0000
    33 N4       -1.9635   -0.0000    0.0578   N.ar      1 RES1   0.0000
    34 H9        1.9432   -0.0000    2.4500   H         1 RES1   0.0000
    35 H10       1.5771    0.0000    4.9221   H         1 RES1   0.0000
    36 H11      -0.7762    0.0000    5.7609   H         1 RES1   0.0000
    37 H12      -2.6344    0.0000    4.1563   H         1 RES1   0.0000
    38 H13      -4.1563    0.0000    2.6344   H         1 RES1   0.0000
    39 H14      -5.7609    0.0000    0.7762   H         1 RES1   0.0000
    40 H15      -4.9221   -0.0000   -1.5771   H         1 RES1   0.0000
    41 H16      -2.4500   -0.0000   -1.9432   H         1 RES1   0.0000
    42 N5        1.9635    0.0578   -0.0000   N.ar      1 RES1   0.0000
    43 C21       2.8705   -0.9419    0.0000   C.ar      1 RES1   0.0000
    44 C22       4.2358   -0.7384   -0.0000   C.ar      1 RES1   0.0000
    45 C23       4.6943    0.5697   -0.0000   C.ar      1 RES1   0.0000
    46 C24       3.7747    1.6208   -0.0000   C.ar      1 RES1   0.0000
    47 C25       2.4031    1.3474   -0.0000   C.ar      1 RES1   0.0000
    48 C26       1.3474    2.4031   -0.0000   C.ar      1 RES1   0.0000
    49 C27       1.6209    3.7747   -0.0000   C.ar      1 RES1   0.0000
    50 C28       0.5697    4.6943   -0.0000   C.ar      1 RES1   0.0000
    51 C29      -0.7384    4.2358    0.0000   C.ar      1 RES1   0.0000
    52 C30      -0.9419    2.8705    0.0000   C.ar      1 RES1   0.0000
    53 N6        0.0578    1.9635   -0.0000   N.ar      1 RES1   0.0000
    54 H17       2.4500   -1.9432    0.0000   H         1 RES1   0.0000
    55 H18       4.9221   -1.5771   -0.0000   H         1 RES1   0.0000
    56 H19       5.7609    0.7762   -0.0000   H         1 RES1   0.0000
    57 H20       4.1563    2.6344   -0.0000   H         1 RES1   0.0000
    58 H21       2.6344    4.1563   -0.0000   H         1 RES1   0.0000
    59 H22       0.7762    5.7609   -0.0000   H         1 RES1   0.0000
    60 H23      -1.5771    4.9221    0.0000   H         1 RES1   0.0000
    61 H24      -1.9432    2.4500    0.0000   H         1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1    13    1
     3     1    22    1
     4     1    33    1
     5     1    42    1
     6     1    53    1
     7     2     3    1
     8     2     7    2
     9     3     4    2
    10     3    14    1
    11     4     5    1
    12     4    15    1
    13     5     6    2
    14     5    16    1
    15     6     7    1
    16     6    17    1
    17     7     8    1
    18     8     9    1
    19     8    13    2
    20     9    10    2
    21     9    18    1
    22    10    11    1
    23    10    19    1
    24    11    12    2
    25    11    20    1
    26    12    13    1
    27    12    21    1
    28    22    23    1
    29    22    27    2
    30    23    24    2
    31    23    34    1
    32    24    25    1
    33    24    35    1
    34    25    26    2
    35    25    36    1
    36    26    27    1
    37    26    37    1
    38    27    28    1
    39    28    29    1
    40    28    33    2
    41    29    30    2
    42    29    38    1
    43    30    31    1
    44    30    39    1
    45    31    32    2
    46    31    40    1
    47    32    33    1
    48    32    41    1
    49    42    43    1
    50    42    47    2
    51    43    44    2
    52    43    54    1
    53    44    45    1
    54    44    55    1
    55    45    46    2
    56    45    56    1
    57    46    47    1
    58    46    57    1
    59    47    48    1
    60    48    49    1
    61    48    53    2
    62    49    50    2
    63    49    58    1
    64    50    51    1
    65    50    59    1
    66    51    52    2
    67    51    60    1
    68    52    53    1
    69    52    61    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       61 GROUP             0 ****  ****    0  
"""


# Functions
class Test02import(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crestPath = shutil.which('crest')
        cls.path = os.path.abspath('.')


    def test_import(self):
        inputDict = {
            "core": {
                "smiles": "[Fe]",
                "coordList": [
                    [2., 0.0, 0.0],
                    [0.0, 2., 0.0],
                    [0.0, 0.0, 2.],
                    [-2., 0.0, 0.0],
                    [0.0, -2., 0.0],
                    [0.0, 0.0, -2.]
                ]
            },
            "ligands": [
                {
                    "smiles": "n1ccccc1-c2ccccn2",
                    "coordList": [[0, 0], [11, 1]]
                },
                {
                    "smiles": "n1ccccc1-c2ccccn2",
                    "coordList": [[0, 2], [11, 3]]
                },
                {
                    "smiles": "n1ccccc1-c2ccccn2",
                    "coordList": [[0, 4], [11, 5]]
                }
            ],
            "parameters": {"crestPath": self.crestPath,
                           "relax": False,
                           "assemble_method":'GFN-FF'}
        }

        # Build complex
        out = arch_bc.build_complex(inputDict)
        keys = out.keys()
        good = False
        for key in keys: # Check for rmsd vs. reference for all generated complexes.
            rmsd_core, rmsd_full, _ = io_align_mol.calc_rmsd(out[key]['mol2string'],refmol)
            if (rmsd_core < 0.1) and (rmsd_full < 0.1):
                good = True
        self.assertEqual(good,True)

if __name__ == "__main__":
    unittest.main()
