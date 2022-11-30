"""
Try building a complex through importing architector.
"""

# Imports
import unittest
import architector.complex_construction as arch_bc
import architector.io_ptable as io_ptable
import os

ligands = ['O','[CH3]-','N','[C-]#[O+]','[Cl-]']
ligands = ['O','[Cl-]']

# Functions
class Test02import(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.abspath('.')

    def test_simple_build(self):
        good = True
        errors = []
        for metal in io_ptable.limited_all_metals:
            for lig in ligands:
                print(metal+lig)
                inputDict = {'core': 
                {'metal': metal, 'coreCN':1}, 
                'ligands': [
                {'smiles': lig, 'coordList': [0]}, 
                ],
                'parameters': {'relax':False,
                'assemble_method':'GFN2-xTB'}, #Conformers to 2 to make sure we get the right one out.
                }

                out = arch_bc.build_complex(inputDict)
                if len(out) < 1:
                    errors.append(metal+lig)
                    good = False

        if good:
            print('Simple Lig Test Good!')
        else:
            print('Simple Lig test failed: ')
            print('Simple Lig Errors: ', errors)
        self.assertEqual(good,True)

if __name__ == "__main__":
    unittest.main()
