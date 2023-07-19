from architector import (build_complex,
                         convert_io_molecule)
import unittest
import os

inp1 = {'core':{'metal':'Fe',
               'coreCN':6},
      'ligands':['water'],
      'parameters':{}}
ref1 = """@<TRIPOS>MOLECULE
octahedral_0_nunpairedes_4_charge_2 Charge: 2 Unpaired_Electrons: 4 XTB_Unpaired_Electrons: 4 XTB_Charge: 2
    19    18     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Fe1       0.0000    0.0000    0.0000   Fe        1 RES1   0.0000
     2 O1       -0.0000    0.0000    2.3471   O.3       1 RES1   0.0000
     3 H1       -0.1211    0.7637    2.9217   H         1 RES1   0.0000
     4 H2        0.1211   -0.7637    2.9217   H         1 RES1   0.0000
     5 O2        0.0000   -0.0000   -2.3471   O.3       1 RES1   0.0000
     6 H3        0.7637   -0.1211   -2.9217   H         1 RES1   0.0000
     7 H4       -0.7637    0.1211   -2.9217   H         1 RES1   0.0000
     8 O3        2.3585    0.0037   -0.0573   O.3       1 RES1   0.0000
     9 H5        2.9439   -0.7580   -0.1089   H         1 RES1   0.0000
    10 H6        2.9331    0.7490    0.1461   H         1 RES1   0.0000
    11 O4       -2.3585   -0.0037   -0.0573   O.3       1 RES1   0.0000
    12 H7       -2.9439    0.7580   -0.1089   H         1 RES1   0.0000
    13 H8       -2.9331   -0.7490    0.1461   H         1 RES1   0.0000
    14 O5        0.0037    2.3585    0.0573   O.3       1 RES1   0.0000
    15 H9       -0.7580    2.9439    0.1089   H         1 RES1   0.0000
    16 H10       0.7490    2.9331   -0.1461   H         1 RES1   0.0000
    17 O6       -0.0037   -2.3585    0.0573   O.3       1 RES1   0.0000
    18 H11       0.7580   -2.9439    0.1089   H         1 RES1   0.0000
    19 H12      -0.7490   -2.9331   -0.1461   H         1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    17    1
     7     2     3    1
     8     2     4    1
     9     5     6    1
    10     5     7    1
    11     8     9    1
    12     8    10    1
    13    11    12    1
    14    11    13    1
    15    14    15    1
    16    14    16    1
    17    17    18    1
    18    17    19    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       19 GROUP             0 ****  ****    0  
"""

# Functions
class Test05_1_FeH2O6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.abspath('.')

    def test_build(self):
        out = build_complex(inp1)
        out_gen = convert_io_molecule(out[list(out.keys())[0]]['mol2string'])
        out_ref = convert_io_molecule(ref1)
        self.assertEqual(out_gen.uhf,out_ref.uhf)
        self.assertEqual(out_gen.charge,out_ref.charge)
        self.assertEqual(out_gen.xtb_uhf,out_ref.xtb_uhf)
        self.assertEqual(out_gen.xtb_charge,out_ref.xtb_charge)
        self.assertCountEqual(out_gen.ase_atoms.get_chemical_symbols(),
                              out_ref.ase_atoms.get_chemical_symbols())

inp2 = {'core':{'metal':'Nd',
               'coreCN':6},
      'ligands':['water'],
      'parameters':{'full_method':'UFF',
                    'assemble_method':'UFF'}}

ref2 = """@<TRIPOS>MOLECULE
hexagonal_planar_0_nunpairedes_0_charge_3 Charge: 3 Unpaired_Electrons: 3 XTB_Unpaired_Electrons: 0 XTB_Charge: 3
    19    18     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Nd1       0.0000    0.0000    0.0000   Nd        1 RES1   0.0000
     2 O1       -0.6793    1.5978    1.1113   O.3       1 RES1   0.0000
     3 H1       -0.2970    1.4532    2.0132   H         1 RES1   0.0000
     4 H2       -1.6491    1.4376    1.2283   H         1 RES1   0.0000
     5 O2        0.7443   -1.7910   -1.2393   O.3       1 RES1   0.0000
     6 H3        1.7320   -1.7288   -1.2083   H         1 RES1   0.0000
     7 H4        0.5264   -2.5830   -0.6868   H         1 RES1   0.0000
     8 O3        2.1318    0.3469    0.7966   O.3       1 RES1   0.0000
     9 H5        2.3542   -0.4799    1.2935   H         1 RES1   0.0000
    10 H6        2.7074    0.3132   -0.0084   H         1 RES1   0.0000
    11 O4       -0.4422   -1.4125    1.7641   O.3       1 RES1   0.0000
    12 H7       -1.2705   -1.0557    2.1711   H         1 RES1   0.0000
    13 H8        0.2645   -1.2185    2.4304   H         1 RES1   0.0000
    14 O5       -2.1293   -0.3473   -0.8030   O.3       1 RES1   0.0000
    15 H9       -1.9932   -0.7561   -1.6942   H         1 RES1   0.0000
    16 H10      -2.4769    0.5584   -0.9977   H         1 RES1   0.0000
    17 O6        0.4379    1.4057   -1.7674   O.3       1 RES1   0.0000
    18 H11      -0.3388    2.0188   -1.7969   H         1 RES1   0.0000
    19 H12       0.3323    0.8337   -2.5690   H         1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    17    1
     7     2     3    1
     8     2     4    1
     9     5     6    1
    10     5     7    1
    11     8     9    1
    12     8    10    1
    13    11    12    1
    14    11    13    1
    15    14    15    1
    16    14    16    1
    17    17    18    1
    18    17    19    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       19 GROUP             0 ****  ****    0  """

# Functions
class Test05_2_NdH2O6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.abspath('.')

    def test_build(self):
        out = build_complex(inp2)
        out_gen = convert_io_molecule(out[list(out.keys())[0]]['mol2string'])
        out_ref = convert_io_molecule(ref2)
        self.assertEqual(out_gen.uhf,out_ref.uhf)
        self.assertEqual(out_gen.charge,out_ref.charge)
        self.assertEqual(out_gen.xtb_uhf,out_ref.xtb_uhf)
        self.assertEqual(out_gen.xtb_charge,out_ref.xtb_charge)
        self.assertCountEqual(out_gen.ase_atoms.get_chemical_symbols(),
                              out_ref.ase_atoms.get_chemical_symbols())

inp3 = {'core':{'metal':'Nd',
               'coreCN':6},
      'ligands':['water'],
      'parameters':{'metal_ox':4,'full_method':'UFF',
                    'assemble_method':'UFF'}}

ref3 = """@<TRIPOS>MOLECULE
hexagonal_planar_0_nunpairedes_0_charge_3 Charge: 4 Unpaired_Electrons: 2 XTB_Unpaired_Electrons: 0 XTB_Charge: 3
    19    18     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Nd1       0.0000    0.0000    0.0000   Nd        1 RES1   0.0000
     2 O1       -0.6793    1.5978    1.1113   O.3       1 RES1   0.0000
     3 H1       -0.5069    2.3883    0.5403   H         1 RES1   0.0000
     4 H2       -0.0348    1.6848    1.8577   H         1 RES1   0.0000
     5 O2        0.7308   -1.8124   -1.2185   O.3       1 RES1   0.0000
     6 H3        1.4567   -1.4634   -1.7944   H         1 RES1   0.0000
     7 H4       -0.0104   -2.0139   -1.8436   H         1 RES1   0.0000
     8 O3        2.0877    0.1952    0.9507   O.3       1 RES1   0.0000
     9 H5        2.6181   -0.5313    0.5361   H         1 RES1   0.0000
    10 H6        1.9652   -0.0934    1.8897   H         1 RES1   0.0000
    11 O4       -0.6470   -1.4204    1.6906   O.3       1 RES1   0.0000
    12 H7       -0.4777   -0.9139    2.5254   H         1 RES1   0.0000
    13 H8        0.0390   -2.1348    1.6907   H         1 RES1   0.0000
    14 O5       -2.0855   -0.1954   -0.9543   O.3       1 RES1   0.0000
    15 H9       -1.9100   -0.2801   -1.9251   H         1 RES1   0.0000
    16 H10      -2.4916    0.7024   -0.8551   H         1 RES1   0.0000
    17 O6        0.6452    1.4200   -1.6959   O.3       1 RES1   0.0000
    18 H11      -0.1955    1.8429   -2.0030   H         1 RES1   0.0000
    19 H12       0.9075    0.8255   -2.4420   H         1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    17    1
     7     2     3    1
     8     2     4    1
     9     5     6    1
    10     5     7    1
    11     8     9    1
    12     8    10    1
    13    11    12    1
    14    11    13    1
    15    14    15    1
    16    14    16    1
    17    17    18    1
    18    17    19    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       19 GROUP             0 ****  ****    0  
"""

# Functions
class Test05_3_NdIVH2O6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.abspath('.')

    def test_build(self):
        out = build_complex(inp3)
        out_gen = convert_io_molecule(out[list(out.keys())[0]]['mol2string'])
        out_ref = convert_io_molecule(ref3)
        self.assertEqual(out_gen.uhf,out_ref.uhf)
        self.assertEqual(out_gen.charge,out_ref.charge)
        self.assertEqual(out_gen.xtb_uhf,out_ref.xtb_uhf)
        self.assertEqual(out_gen.xtb_charge,out_ref.xtb_charge)
        self.assertCountEqual(out_gen.ase_atoms.get_chemical_symbols(),
                              out_ref.ase_atoms.get_chemical_symbols())


inp4 = {'core':{'metal':'U',
               'coreCN':6},
      'ligands':['water'],
      'parameters':{'metal_ox':4,'full_method':'UFF',
                    'assemble_method':'UFF'}}

ref4 = """@<TRIPOS>MOLECULE
hexagonal_planar_0_nunpairedes_0_charge_3 Charge: 4 Unpaired_Electrons: 2 XTB_Unpaired_Electrons: 0 XTB_Charge: 3
    19    18     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 U1        0.0000    0.0000    0.0000   U         1 RES1   0.0000
     2 O1       -0.5897    1.3870    0.9647   O.3       1 RES1   0.0000
     3 H1       -0.8450    0.9924    1.8363   H         1 RES1   0.0000
     4 H2       -1.4438    1.6805    0.5619   H         1 RES1   0.0000
     5 O2        0.7584   -1.7629   -1.2721   O.3       1 RES1   0.0000
     6 H3        1.7374   -1.7855   -1.1285   H         1 RES1   0.0000
     7 H4        0.4193   -2.5693   -0.8092   H         1 RES1   0.0000
     8 O3        2.0660    0.1920    0.9996   O.3       1 RES1   0.0000
     9 H5        2.4716   -0.7075    0.9216   H         1 RES1   0.0000
    10 H6        1.8716    0.2907    1.9655   H         1 RES1   0.0000
    11 O4       -0.6554   -1.5056    1.6247   O.3       1 RES1   0.0000
    12 H7       -1.6035   -1.2913    1.8133   H         1 RES1   0.0000
    13 H8       -0.1660   -1.2400    2.4444   H         1 RES1   0.0000
    14 O5       -2.0634   -0.2064   -1.0116   O.3       1 RES1   0.0000
    15 H9       -1.9117   -0.8354   -1.7607   H         1 RES1   0.0000
    16 H10      -2.2255    0.6651   -1.4528   H         1 RES1   0.0000
    17 O6        0.6647    1.4655   -1.6433   O.3       1 RES1   0.0000
    18 H11       1.6114    1.2312   -1.8137   H         1 RES1   0.0000
    19 H12       0.1857    1.1627   -2.4553   H         1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    17    1
     7     2     3    1
     8     2     4    1
     9     5     6    1
    10     5     7    1
    11     8     9    1
    12     8    10    1
    13    11    12    1
    14    11    13    1
    15    14    15    1
    16    14    16    1
    17    17    18    1
    18    17    19    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       19 GROUP             0 ****  ****    0"""

# Functions
class Test05_4_UIVH2O6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.abspath('.')

    def test_build(self):
        out = build_complex(inp4)
        out_gen = convert_io_molecule(out[list(out.keys())[0]]['mol2string'])
        out_ref = convert_io_molecule(ref4)
        self.assertEqual(out_gen.uhf,out_ref.uhf)
        self.assertEqual(out_gen.charge,out_ref.charge)
        self.assertEqual(out_gen.xtb_uhf,out_ref.xtb_uhf)
        self.assertEqual(out_gen.xtb_charge,out_ref.xtb_charge)
        self.assertCountEqual(out_gen.ase_atoms.get_chemical_symbols(),
                              out_ref.ase_atoms.get_chemical_symbols())


inp5 =  {'core':{'metal':'U',
               'coreCN':6},
      'ligands':['water']+['oxo']*2,
      'parameters':{'metal_ox':5,'full_method':'UFF',
                    'assemble_method':'UFF'}}

ref5 = """@<TRIPOS>MOLECULE
octahedral_0_nunpairedes_0_charge_-1 Charge: 1 Unpaired_Electrons: 1 XTB_Unpaired_Electrons: 0 XTB_Charge: -1
    15    14     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 U1        0.0000    0.0000    0.0000   U         1 RES1   0.0000
     2 O1        1.7895    0.0000    0.0000   O.3       1 RES1   0.0000
     3 H1        2.1503   -0.3672   -0.8468   H         1 RES1   0.0000
     4 H2        2.1225   -0.6098    0.7072   H         1 RES1   0.0000
     5 O2       -1.7895    0.0000    0.0000   O.3       1 RES1   0.0000
     6 H3       -2.1205   -0.6180    0.7009   H         1 RES1   0.0000
     7 H4       -2.1497   -0.3607   -0.8498   H         1 RES1   0.0000
     8 O3        0.0000    1.7895    0.0000   O.3       1 RES1   0.0000
     9 H5       -0.7800    2.1107   -0.5201   H         1 RES1   0.0000
    10 H6        0.7800    2.1100   -0.5206   H         1 RES1   0.0000
    11 O4        0.0000    0.0000   -1.7895   O.3       1 RES1   0.0000
    12 O5        0.0000    0.0000    1.7895   O.3       1 RES1   0.0000
    13 O6        0.0000   -1.7895    0.0000   O.3       1 RES1   0.0000
    14 H7        0.7852   -2.1395   -0.4914   H         1 RES1   0.0000
    15 H8       -0.7629   -2.1331   -0.5298   H         1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    12    1
     6     1    13    1
     7     2     3    1
     8     2     4    1
     9     5     6    1
    10     5     7    1
    11     8     9    1
    12     8    10    1
    13    13    14    1
    14    13    15    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       15 GROUP             0 ****  ****    0 """

# Functions
class Test05_5_UVO2H2O4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.abspath('.')

    def test_build(self):
        out = build_complex(inp5)
        out_gen = convert_io_molecule(out[list(out.keys())[0]]['mol2string'])
        out_ref = convert_io_molecule(ref5)
        self.assertEqual(out_gen.uhf,out_ref.uhf)
        self.assertEqual(out_gen.charge,out_ref.charge)
        self.assertEqual(out_gen.xtb_uhf,out_ref.xtb_uhf)
        self.assertEqual(out_gen.xtb_charge,out_ref.xtb_charge)
        self.assertCountEqual(out_gen.ase_atoms.get_chemical_symbols(),
                              out_ref.ase_atoms.get_chemical_symbols())

if __name__ == "__main__":
    unittest.main()