from architector import (build_complex,
                         convert_io_molecule)
import architector.io_align_mol as io_align_mol
import unittest
import os

inp1 = {'core':{'metal':'Am','coreCN':7},
        'ligands':['oxo','oxo',{'smiles':'O[O-]','coordList':[0,1],'ligType':'edge_bi_cis'}],
        'parameters':{'assemble_method':'UFF',
                    'full_method':'UFF',
                    'metal_ox':6}}

ref1 = """@<TRIPOS>MOLECULE
pentagonal_bipyramidal_0_nunpairedes_0_charge_-2 Charge: 1 Unpaired_Electrons: 3 XTB_Unpaired_Electrons: 0 XTB_Charge: -2
    15    15     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Am1       0.0000    0.0000    0.0000   Am        1 RES1   0.0000
     2 O1       -1.3578   -1.9465    0.0000   O.3       1 RES1   0.0000
     3 O2       -0.0674   -2.3048    0.0000   O.co2     1 RES1   0.0000
     4 H1       -1.6446   -2.0462    0.9425   H         1 RES1   0.0000
     5 O3        2.2585    0.0000    0.0000   O.3       1 RES1   0.0000
     6 H2        2.5779    0.5037    0.7898   H         1 RES1   0.0000
     7 H3        2.5767    0.4924   -0.7973   H         1 RES1   0.0000
     8 O4        0.6979    2.1480    0.0000   O.3       1 RES1   0.0000
     9 H4        1.2759    2.2961   -0.7895   H         1 RES1   0.0000
    10 H5        1.2637    2.2987    0.7979   H         1 RES1   0.0000
    11 O5       -1.8269    1.3279    0.0000   O.3       1 RES1   0.0000
    12 H6       -1.9471    1.6088    0.9420   H         1 RES1   0.0000
    13 H7       -2.5826    0.7119   -0.1730   H         1 RES1   0.0000
    14 O6        0.0000    0.0000   -1.7675   O.3       1 RES1   0.0000
    15 O7        0.0000    0.0000    1.7675   O.3       1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     3    1
     3     1     5    1
     4     1     8    1
     5     1    11    1
     6     1    14    1
     7     1    15    1
     8     2     3    1
     9     2     4    1
    10     5     6    1
    11     5     7    1
    12     8     9    1
    13     8    10    1
    14    11    12    1
    15    11    13    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       15 GROUP             0 ****  ****    0  
"""

class Test07_1_AmO2Peroxo_water3_Edge(unittest.TestCase):
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
        good = False
        keys = list(out.keys())
        for key in keys: # Check for rmsd vs. reference for all generated complexes.
            rmsd_core, rmsd_full, _ = io_align_mol.calc_rmsd(out[key]['mol2string'],out_ref)
            if (rmsd_core < 1.0) and (rmsd_full < 1.0): # Tested a few times - may have to raise.
                good = True
        self.assertEqual(good,True)
            
inp2 ={'core':{'metal':'Pt','coreCN':5},
       'ligands':['chloride']*3 +[{'smiles':'C=C','coordList':[0,1]}],
       'parameters':{'skip_duplicate_tests':True}}

ref2 = """@<TRIPOS>MOLECULE
trigonal_bipyramidal_0_nunpairedes_0_charge_-1 Charge: -1 Unpaired_Electrons: 0 XTB_Unpaired_Electrons: 0 XTB_Charge: -1
    10    10     1     0     0
SMALL
NoCharges
****
Generated from Architector

@<TRIPOS>ATOM
     1 Pt1      -0.0573    0.0232    0.2663   Pt        1 RES1   0.0000
     2 C1        1.6833    0.0102   -0.9718   C.2       1 RES1   0.0000
     3 C2        0.5551    0.0083   -1.7792   C.2       1 RES1   0.0000
     4 H1        2.2426    0.9298   -0.8264   H         1 RES1   0.0000
     5 H2        2.2394   -0.9084   -0.8193   H         1 RES1   0.0000
     6 H3        0.2421    0.9291   -2.2640   H         1 RES1   0.0000
     7 H4        0.2249   -0.9114   -2.2513   H         1 RES1   0.0000
     8 Cl1      -1.3528    0.0301    2.0888   Cl        1 RES1   0.0000
     9 Cl2      -0.0844   -2.2922    0.3109   Cl        1 RES1   0.0000
    10 Cl3      -0.0582    2.3468    0.2499   Cl        1 RES1   0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     1     3    1
     3     1     8    1
     4     1     9    1
     5     1    10    1
     6     2     3    2
     7     2     4    1
     8     2     5    1
     9     3     6    1
    10     3     7    1
@<TRIPOS>SUBSTRUCTURE
     1 RES1       10 GROUP             0 ****  ****    0   """

class Test07_2_PtCl3Ethylene_Edge(unittest.TestCase):
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
        good = False
        keys = list(out.keys())
        for key in keys: # Check for rmsd vs. reference for all generated complexes.
            rmsd_core, rmsd_full, _ = io_align_mol.calc_rmsd(out[key]['mol2string'],out_ref)
            if (rmsd_core < 0.3) and (rmsd_full < 1):
                good = True
        self.assertEqual(good,True)