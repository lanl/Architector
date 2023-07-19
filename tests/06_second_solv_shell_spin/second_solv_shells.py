from architector import (build_complex,
                         convert_io_molecule)
import unittest
import numpy as np
import os

inp1 = {'core':{'metal':'Cu',
               'coreCN':4},
      'ligands':['water'],
      'parameters':{'add_secondary_shell_species':True,
                   'species_list':['water']*3}}

ref1 = """@<TRIPOS>MOLECULE
Mol_Plus_Species_Example_Energy Charge: 2 Unpaired_Electrons: 1 XTB_Unpaired_Electrons: 1 XTB_Charge: 2
    22    18     4     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Cu1       0.4972   -0.8093    0.0186   Cu        1 RES1   0.0000
     2 O1        2.5434   -0.7627    0.2329   O.3       1 RES1   0.0000
     3 H1        2.9360   -0.2962   -0.5200   H         1 RES1   0.0000
     4 H2        2.8906   -0.3436    1.0354   H         1 RES1   0.0000
     5 O2       -1.4820   -0.4865    0.2854   O.3       1 RES1   0.0000
     6 H3       -1.8136    0.2573   -0.2773   H         1 RES1   0.0000
     7 H4       -2.1139   -1.2150    0.2632   H         1 RES1   0.0000
     8 O3        0.5660    1.1674    0.0936   O.3       1 RES1   0.0000
     9 H5       -0.0601    1.5111   -0.5825   H         1 RES1   0.0000
    10 H6        0.2189    1.5874    0.9486   H         1 RES1   0.0000
    11 O4        0.6517   -2.8248    0.4913   O.3       1 RES1   0.0000
    12 H7        0.3792   -3.1167    1.3727   H         1 RES1   0.0000
    13 H8        1.5417   -3.1746    0.3413   H         1 RES1   0.0000
    14 O5       -1.7308    1.7451   -1.2005   O.3       2 RES2   0.0000
    15 H9       -2.0979    2.4927   -0.7126   H         2 RES2   0.0000
    16 H10      -1.9773    1.8801   -2.1223   H         2 RES2   0.0000
    17 O6       -0.5480    2.4133    2.0293   O.3       3 RES3   0.0000
    18 H11      -0.2333    3.3001    2.2367   H         3 RES3   0.0000
    19 H12      -0.9383    2.0637    2.8355   H         3 RES3   0.0000
    20 O7        0.3547   -1.4882   -2.0409   O.3       4 RES4   0.0000
    21 H13      -0.5064   -1.6768   -2.4311   H         4 RES4   0.0000
    22 H14       0.9223   -2.2237   -2.2974   H         4 RES4   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     2     3    1
     6     2     4    1
     7     5     6    1
     8     5     7    1
     9     8     9    1
    10     8    10    1
    11    11    12    1
    12    11    13    1
    13    14    15    1
    14    14    16    1
    15    17    18    1
    16    17    19    1
    17    20    21    1
    18    20    22    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       13 GROUP             0 ****  ****    0  
     2 RES2        3 GROUP             0 ****  ****    0  
     3 RES3        3 GROUP             0 ****  ****    0  
     4 RES4        3 GROUP             0 ****  ****    0   
"""

class Test06_1_CuH2O4_water3(unittest.TestCase):
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
        for comp in range(4): # Check that all components are equal length
            self.assertEqual(out_gen.find_component_indices(component=comp).shape[0],
                             out_ref.find_component_indices(component=comp).shape[0]
                             )
            
inp2 ={'core':{'metal':'La',
               'coreCN':8},
      'ligands':['water'],
      'parameters':{'add_secondary_shell_species':True,
                    'species_list':['nitrate_bi']*3,
                    'relax':False,
                    'return_only_1':True,
                    'species_relax':False,
                    'species_intermediate_relax':False}}

ref2 = """@<TRIPOS>MOLECULE
Mol_Plus_Species_Example_Energy Charge: 0 Unpaired_Electrons: 0 XTB_Unpaired_Electrons: 0 XTB_Charge: 0
    37    33     4     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 La1       0.0109   -0.2118    0.1394   La        1 RES1   0.0000
     2 O1       -1.7688   -1.0185    1.8299   O.3       1 RES1   0.0000
     3 H1       -2.0688   -1.9472    1.9974   H         1 RES1   0.0000
     4 H2       -2.3039   -0.4682    2.4557   H         1 RES1   0.0000
     5 O2       -2.0696   -1.6769   -0.3086   O.3       1 RES1   0.0000
     6 H3       -2.1246   -2.6656   -0.3157   H         1 RES1   0.0000
     7 H4       -2.9909   -1.3757   -0.5116   H         1 RES1   0.0000
     8 O3       -0.7428    0.6923   -2.1607   O.3       1 RES1   0.0000
     9 H5       -1.3371    1.4628   -2.3441   H         1 RES1   0.0000
    10 H6       -0.5020    0.3460   -3.0567   H         1 RES1   0.0000
    11 O4       -1.1661    2.0816   -0.0365   O.3       1 RES1   0.0000
    12 H7       -0.7452    2.9759   -0.0964   H         1 RES1   0.0000
    13 H8       -2.1393    2.2634   -0.0591   H         1 RES1   0.0000
    14 O5        0.6785   -1.4665   -2.0184   O.3       1 RES1   0.0000
    15 H9        1.1372   -2.2836   -1.6984   H         1 RES1   0.0000
    16 H10       1.4003   -0.9150   -2.4128   H         1 RES1   0.0000
    17 O6        1.9233   -1.9483    0.0836   O.3       1 RES1   0.0000
    18 H11       1.9367   -2.8487    0.4955   H         1 RES1   0.0000
    19 H12       2.8072   -1.8627   -0.3545   H         1 RES1   0.0000
    20 O7        1.6451   -0.5525    2.1115   O.3       1 RES1   0.0000
    21 H13       2.0095   -1.4063    2.4562   H         1 RES1   0.0000
    22 H14       2.0476    0.1414    2.6921   H         1 RES1   0.0000
    23 O8        0.6632    1.7451    1.6954   O.3       1 RES1   0.0000
    24 H15       1.5689    1.9950    2.0080   H         1 RES1   0.0000
    25 H16       0.0634    2.4133    2.1130   H         1 RES1   0.0000
    26 N1       -2.3023    2.3120    2.1491   N.pl3     2 RES2   0.0000
    27 O9       -3.0462    1.3398    2.0799   O.2       2 RES2   0.0000
    28 O10      -1.9983    2.8565    3.3690   O.co2     2 RES2   0.0000
    29 O11      -1.8168    2.8141    1.1218   O.co2     2 RES2   0.0000
    30 N2        0.0424    3.2187   -2.1269   N.pl3     3 RES3   0.0000
    31 O12      -0.7016    2.2465   -2.1961   O.2       3 RES3   0.0000
    32 O13       0.3464    3.7632   -0.9071   O.co2     3 RES3   0.0000
    33 O14       0.5279    3.7208   -3.1542   O.co2     3 RES3   0.0000
    34 N3        2.7424   -3.9540   -1.2906   N.pl3     4 RES4   0.0000
    35 O15       1.9984   -4.9262   -1.3598   O.2       4 RES4   0.0000
    36 O16       3.0464   -3.4095   -0.0707   O.co2     4 RES4   0.0000
    37 O17       3.2279   -3.4519   -2.3179   O.co2     4 RES4   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    17    1
     7     1    20    1
     8     1    23    1
     9     2     3    1
    10     2     4    1
    11     5     6    1
    12     5     7    1
    13     8     9    1
    14     8    10    1
    15    11    12    1
    16    11    13    1
    17    14    15    1
    18    14    16    1
    19    17    18    1
    20    17    19    1
    21    20    21    1
    22    20    22    1
    23    23    24    1
    24    23    25    1
    25    26    27    2
    26    26    28    1
    27    26    29    1
    28    30    31    2
    29    30    32    1
    30    30    33    1
    31    34    35    2
    32    34    36    1
    33    34    37    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       25 GROUP             0 ****  ****    0  
     2 RES2        4 GROUP             0 ****  ****    0  
     3 RES3        4 GROUP             0 ****  ****    0  
     4 RES4        4 GROUP             0 ****  ****    0  """

class Test06_2_LaH2O8_nitrate3(unittest.TestCase):
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
        for comp in range(4): # Check that all components are equal length
            self.assertEqual(out_gen.find_component_indices(component=comp).shape[0],
                             out_ref.find_component_indices(component=comp).shape[0]
                             )
            

inp3 ={'core':{'metal':'Am',
               'coreCN':8},
      'ligands':['water'],
      'parameters':{'add_secondary_shell_species':True,
                    'species_list':['nitrate_bi']*3,
                    'relax':False,
                    'return_only_1':True,
                    'species_relax':False,
                    'species_intermediate_relax':False}}

ref3 = """@<TRIPOS>MOLECULE
Mol_Plus_Species_Example_Energy Charge: 0 Unpaired_Electrons: 6 XTB_Unpaired_Electrons: 0 XTB_Charge: 0
    37    33     4     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Am1       0.5091   -0.0093   -0.1325   Am        1 RES1   0.0000
     2 O1       -0.7083   -0.5611    1.0239   O.3       1 RES1   0.0000
     3 H1       -1.0955   -1.4685    1.1099   H         1 RES1   0.0000
     4 H2       -1.1563   -0.0322    1.7312   H         1 RES1   0.0000
     5 O2       -0.9141   -1.0115   -0.4389   O.3       1 RES1   0.0000
     6 H3       -1.8664   -0.7478   -0.3749   H         1 RES1   0.0000
     7 H4       -0.9380   -1.9628   -0.7132   H         1 RES1   0.0000
     8 O3       -0.0064    0.6091   -1.7059   O.3       1 RES1   0.0000
     9 H5       -0.1624    0.0951   -2.5378   H         1 RES1   0.0000
    10 H6       -0.2041    1.5474   -1.9533   H         1 RES1   0.0000
    11 O4       -0.2960    1.5595   -0.2528   O.3       1 RES1   0.0000
    12 H7        0.0361    2.3787   -0.6992   H         1 RES1   0.0000
    13 H8       -1.1805    1.8165    0.1110   H         1 RES1   0.0000
    14 O5        0.9658   -0.8676   -1.6086   O.3       1 RES1   0.0000
    15 H9        1.7492   -0.6987   -2.1902   H         1 RES1   0.0000
    16 H10       0.4956   -1.6251   -2.0394   H         1 RES1   0.0000
    17 O6        1.8173   -1.1972   -0.1707   O.3       1 RES1   0.0000
    18 H11       1.9225   -1.9633   -0.7892   H         1 RES1   0.0000
    19 H12       2.6094   -1.2458    0.4216   H         1 RES1   0.0000
    20 O7        1.6270   -0.2424    1.2165   O.3       1 RES1   0.0000
    21 H13       2.0050   -1.0947    1.5503   H         1 RES1   0.0000
    22 H14       2.0159    0.4500    1.8081   H         1 RES1   0.0000
    23 O8        0.9553    1.3293    0.9319   O.3       1 RES1   0.0000
    24 H15       0.3969    1.7660    1.6234   H         1 RES1   0.0000
    25 H16       1.8197    1.8108    0.9706   H         1 RES1   0.0000
    26 N1        0.3153   -3.7230   -1.1366   N.pl3     2 RES2   0.0000
    27 O9        0.6846   -3.6984   -2.3056   O.2       2 RES2   0.0000
    28 O10       1.1207   -4.2748   -0.1754   O.co2     2 RES2   0.0000
    29 O11      -0.7862   -3.2515   -0.8089   O.co2     2 RES2   0.0000
    30 N2        0.3153    3.8804    0.8393   N.pl3     3 RES3   0.0000
    31 O12       0.6846    3.9051   -0.3297   O.2       3 RES3   0.0000
    32 O13       1.1207    3.3287    1.8004   O.co2     3 RES3   0.0000
    33 O14      -0.7862    4.3519    1.1669   O.co2     3 RES3   0.0000
    34 N3       -3.2847    0.2259    0.9847   N.pl3     4 RES4   0.0000
    35 O15      -2.9154    0.2505   -0.1843   O.2       4 RES4   0.0000
    36 O16      -2.4793   -0.3259    1.9459   O.co2     4 RES4   0.0000
    37 O17      -4.3862    0.6974    1.3124   O.co2     4 RES4   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    17    1
     7     1    20    1
     8     1    23    1
     9     2     3    1
    10     2     4    1
    11     5     6    1
    12     5     7    1
    13     8     9    1
    14     8    10    1
    15    11    12    1
    16    11    13    1
    17    14    15    1
    18    14    16    1
    19    17    18    1
    20    17    19    1
    21    20    21    1
    22    20    22    1
    23    23    24    1
    24    23    25    1
    25    26    27    2
    26    26    28    1
    27    26    29    1
    28    30    31    2
    29    30    32    1
    30    30    33    1
    31    34    35    2
    32    34    36    1
    33    34    37    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       25 GROUP             0 ****  ****    0  
     2 RES2        4 GROUP             0 ****  ****    0  
     3 RES3        4 GROUP             0 ****  ****    0  
     4 RES4        4 GROUP             0 ****  ****    0  """


class Test06_3_AmH2O8_nitrate3(unittest.TestCase):
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
        for comp in range(4): # Check that all components are equal length
            self.assertCountEqual(np.array(out_gen.ase_atoms.get_chemical_symbols()
                                           )[out_gen.find_component_indices(component=comp)].tolist(),
                        np.array(out_ref.ase_atoms.get_chemical_symbols()
                                           )[out_gen.find_component_indices(component=comp)].tolist())
            

inp4 ={'core':{'metal':'Am',
               'coreCN':6},
      'ligands':['oxo']*2,
      'parameters':{'add_secondary_shell_species':True,
                    'metal_ox':6,
                    'species_list':['water']*2,
                    'relax':False,
                    'return_only_1':True,
                    'species_relax':False,
                    'species_intermediate_relax':False}}

ref4 = """@<TRIPOS>MOLECULE
Mol_Plus_Species_Example_Energy Charge: 2 Unpaired_Electrons: 3 XTB_Unpaired_Electrons: 0 XTB_Charge: -1
    21    18     3     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Am1      -0.0997   -0.3069   -0.1069   Am        1 RES1   0.0000
     2 O1        1.6678   -0.3069   -0.1069   O.3       1 RES1   0.0000
     3 H1        2.2013   -0.3069    0.7281   H         1 RES1   0.0000
     4 H2        2.2013   -0.3069   -0.9419   H         1 RES1   0.0000
     5 O2       -1.8672   -0.3069   -0.1069   O.3       1 RES1   0.0000
     6 H3       -2.4007   -0.3069    0.7281   H         1 RES1   0.0000
     7 H4       -2.4007   -0.3069   -0.9419   H         1 RES1   0.0000
     8 O3       -0.0997    1.4606   -0.1069   O.3       1 RES1   0.0000
     9 H5        0.7352    1.9942   -0.1069   H         1 RES1   0.0000
    10 H6       -0.9346    1.9942   -0.1069   H         1 RES1   0.0000
    11 O4       -0.0997   -2.0744   -0.1069   O.3       1 RES1   0.0000
    12 H7       -0.9346   -2.6080   -0.1069   H         1 RES1   0.0000
    13 H8        0.7352   -2.6080   -0.1069   H         1 RES1   0.0000
    14 O5       -0.0997   -0.3069   -1.8744   O.3       1 RES1   0.0000
    15 O6       -0.0997   -0.3069    1.6606   O.3       1 RES1   0.0000
    16 O7        0.4802    1.5350   -3.2118   O.3       2 RES2   0.0000
    17 H9        0.0959    0.7381   -3.6779   H         2 RES2   0.0000
    18 H10      -0.2783    1.9033   -2.7339   H         2 RES2   0.0000
    19 O8        0.7802    0.2850    3.7382   O.3       3 RES3   0.0000
    20 H11       0.3959   -0.5119    3.2721   H         3 RES3   0.0000
    21 H12       0.0217    0.6533    4.2161   H         3 RES3   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    15    1
     7     2     3    1
     8     2     4    1
     9     5     6    1
    10     5     7    1
    11     8     9    1
    12     8    10    1
    13    11    12    1
    14    11    13    1
    15    16    17    1
    16    16    18    1
    17    19    20    1
    18    19    21    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       15 GROUP             0 ****  ****    0  
     2 RES2        3 GROUP             0 ****  ****    0  
     3 RES3        3 GROUP             0 ****  ****    0 """


class Test06_4_AmVIO2H2O4_water2(unittest.TestCase):
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
        for comp in range(3): # Check that all components are equal length
            self.assertCountEqual(np.array(out_gen.ase_atoms.get_chemical_symbols()
                                           )[out_gen.find_component_indices(component=comp)].tolist(),
                        np.array(out_ref.ase_atoms.get_chemical_symbols()
                                           )[out_gen.find_component_indices(component=comp)].tolist())
            
inp5 ={'core':{'metal':'Am',
               'coreCN':6},
      'ligands':['oxo']*2,
      'parameters':{'add_secondary_shell_species':True,
                    'metal_ox':6,
                    'species_list':[ref1],
                    'species_relax':False,
                    'species_intermediate_relax':False
                   }}

ref5 = """@<TRIPOS>MOLECULE
Mol_Plus_Species_Example_Energy Charge: 4 Unpaired_Electrons: 4 XTB_Unpaired_Electrons: 1 XTB_Charge: 1
    37    32     5     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Am1       0.0601   -0.4751   -3.5075   Am        1 RES1   0.0000
     2 O1        1.8276   -0.4751   -3.5075   O.3       1 RES1   0.0000
     3 H1        2.3611   -0.4751   -4.3425   H         1 RES1   0.0000
     4 H2        2.3611   -0.4751   -2.6725   H         1 RES1   0.0000
     5 O2       -1.7074   -0.4751   -3.5075   O.3       1 RES1   0.0000
     6 H3       -2.2409   -0.4751   -4.3425   H         1 RES1   0.0000
     7 H4       -2.2409   -0.4751   -2.6725   H         1 RES1   0.0000
     8 O3        0.0601    1.2924   -3.5075   O.3       1 RES1   0.0000
     9 H5       -0.7748    1.8260   -3.5075   H         1 RES1   0.0000
    10 H6        0.8950    1.8260   -3.5075   H         1 RES1   0.0000
    11 O4        0.0601   -2.2426   -3.5075   O.3       1 RES1   0.0000
    12 H7        0.8950   -2.7762   -3.5075   H         1 RES1   0.0000
    13 H8       -0.7748   -2.7762   -3.5075   H         1 RES1   0.0000
    14 O5        0.0601   -0.4751   -5.2750   O.3       1 RES1   0.0000
    15 O6        0.0601   -0.4751   -1.7400   O.3       1 RES1   0.0000
    16 Cu1       0.1501   -0.3517    3.0314   Cu        2 RES2   0.0000
    17 O7       -0.9937   -2.0609    3.1054   O.3       2 RES2   0.0000
    18 H9       -0.9115   -2.5405    2.2676   H         2 RES2   0.0000
    19 H10      -1.9420   -1.9468    3.2721   H         2 RES2   0.0000
    20 O8        0.7281    1.5859    2.9664   O.3       2 RES2   0.0000
    21 H11       0.8712    1.8972    2.0377   H         2 RES2   0.0000
    22 H12       1.4388    1.9138    3.5302   H         2 RES2   0.0000
    23 O9       -1.0083    0.2128    1.5290   O.3       2 RES2   0.0000
    24 H13      -0.4297    0.6210    0.8465   H         2 RES2   0.0000
    25 H14      -1.6644    0.9577    1.7354   H         2 RES2   0.0000
    26 O10       0.8442   -0.9245    4.9022   O.3       2 RES2   0.0000
    27 H15       0.5211   -0.4509    5.6817   H         2 RES2   0.0000
    28 H16       0.7162   -1.8687    5.0730   H         2 RES2   0.0000
    29 O11       0.6634    1.9369    0.2976   O.3       3 RES3   0.0000
    30 H17       0.0933    2.6644    0.0194   H         3 RES3   0.0000
    31 H18       1.3457    1.8581   -0.3784   H         3 RES3   0.0000
    32 O12      -2.4984    2.2773    1.7688   O.3       4 RES4   0.0000
    33 H19      -3.2730    2.3441    1.1996   H         4 RES4   0.0000
    34 H20      -2.6821    2.8097    2.5481   H         4 RES4   0.0000
    35 O13       2.0136   -1.1838    2.2846   O.3       5 RES5   0.0000
    36 H21       2.7946   -0.6252    2.1993   H         5 RES5   0.0000
    37 H22       2.3218   -1.9996    2.6949   H         5 RES5   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     5    1
     3     1     8    1
     4     1    11    1
     5     1    14    1
     6     1    15    1
     7     2     3    1
     8     2     4    1
     9     5     6    1
    10     5     7    1
    11     8     9    1
    12     8    10    1
    13    11    12    1
    14    11    13    1
    15    16    17    1
    16    16    20    1
    17    16    23    1
    18    16    26    1
    19    17    18    1
    20    17    19    1
    21    20    21    1
    22    20    22    1
    23    23    24    1
    24    23    25    1
    25    26    27    1
    26    26    28    1
    27    29    30    1
    28    29    31    1
    29    32    33    1
    30    32    34    1
    31    35    36    1
    32    35    37    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       15 GROUP             0 ****  ****    0  
     2 RES2       13 GROUP             0 ****  ****    0  
     3 RES3        3 GROUP             0 ****  ****    0  
     4 RES4        3 GROUP             0 ****  ****    0  
     5 RES5        3 GROUP             0 ****  ****    0  """

class Test06_5_AmVIO2H2O4_CuH2O7(unittest.TestCase):
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
        for comp in range(5): # Check that all components are equal length
            self.assertCountEqual(np.array(out_gen.ase_atoms.get_chemical_symbols()
                                           )[out_gen.find_component_indices(component=comp)].tolist(),
                        np.array(out_ref.ase_atoms.get_chemical_symbols()
                                           )[out_gen.find_component_indices(component=comp)].tolist())