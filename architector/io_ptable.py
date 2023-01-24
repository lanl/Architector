""" 
Essentially data scripts for handling periodic table properties.
Needed for handling few cases not readily handled elswhere and for architector-specific tunings.

Developed by Michael Taylor and Dan Burril.
"""

import numpy as np
import copy 

elements = ('X',
        "H",                                                                                                 "He",
        "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
        "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
        "Cs", "Ba", 
                    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                        "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra",
                    "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
        )  

# Data from Dalton Trans., 2013, 42, 8617-8636 -> Assigning Pm to average of neighbors = 2.92, Fr to 3.28, Ra to 3.10
rvdw = (0.00,
        1.20,                                                                                                 1.43,
        2.12, 1.98,                                                             1.91, 1.77, 1.66, 1.50, 1.46, 1.58,
        2.50, 2.51,                                                             2.25, 2.19, 1.90, 1.89, 1.82, 1.83,
        2.73, 2.62, 2.58, 2.46, 2.42, 2.45, 2.45, 2.44, 2.40, 2.40, 2.38, 2.39, 2.32, 2.29, 1.88, 1.82, 1.86, 2.25,
        3.21, 2.84, 2.75, 2.52, 2.56, 2.45, 2.44, 2.46, 2.44, 2.15, 2.53, 2.49, 2.43, 2.42, 2.47, 1.99, 2.04, 2.06,
        3.48, 3.03, 
                    2.98, 2.88, 2.92, 2.95, 2.92 , 2.90, 2.87, 2.83, 2.79, 2.87, 2.81, 2.83, 2.79, 2.80, 2.74,
                        2.63, 2.53, 2.57, 2.49, 2.48, 2.41, 2.29, 2.32, 2.45, 2.47, 2.60, 2.54, "Po", "At", "Rn",
        3.28, 3.10,
                    2.80, 2.93, 2.88, 2.71, 2.82, 2.81, 2.83, 3.05, 3.40, 3.05, 2.70, "Fm", "Md", "No", "Lr",
                        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
        )


# Data from J. Phys. Chem. A 2015, 119, 2326-2337
# single bond only
rcov1 = (0.00,
        0.32,                                                                                                 0.46,
        1.33, 1.02,                                                             0.85, 0.75, 0.71, 0.63, 0.64, 0.67,
        1.55, 1.39,                                                             1.26, 1.16, 1.11, 1.03, 0.99, 0.96,
        1.96, 1.71, 1.48, 1.36, 1.34, 1.22, 1.19, 1.16, 1.11, 1.10, 1.12, 1.18, 1.24, 1.21, 1.21, 1.16, 1.14, 1.17,
        2.10, 1.85, 1.63, 1.54, 1.47, 1.38, 1.28, 1.25, 1.25, 1.20, 1.28, 1.36, 1.42, 1.40, 1.40, 1.36, 1.33, 1.31,
        2.32, 1.96, 
                    1.80, 1.63, 1.76, 1.74, 1.73, 1.72, 1.68, 1.69, 1.68, 1.67, 1.66, 1.65, 1.64, 1.70, 1.62,
                        1.52, 1.46, 1.37, 1.31, 1.29, 1.22, 1.23, 1.24, 1.33, 1.44, 1.44, 1.51, 1.45, 1.47, 1.42,
        2.23, 2.01,
                    1.86, 1.75, 1.69, 1.70, 1.71, 1.72, 1.66, 1.66, 1.68, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61,
                        1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57
)

# Used mostly for charge on ligand detection. Metal valence electron counts shouldn't be referenced anywhere (untrustworthy)
valence_electrons = (0.00,
    1,                                                    2,
    1, 2,                                  3, 4, 5, 6, 7, 8,
    1, 2,                                  3, 4, 5, 6, 7, 8,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8,
    1, 2, 
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 3, # Lu - 3 valence elctrons
                    4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, # Hg - 2 valence electrons
    1, 2,
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 3, # Lr - 3 valence elctrons
                    4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, # Cn - 2 valence electrons
)

masses = (0.00, # Masses
    1.0080,                                                                                                                                                                   4.0026,
    6.9400,   9.0122,                                                                                                      10.8100,  12.0110,  14.0070,  15.9990,  18.9980,  20.1800,
    22.9900,  24.3050,                                                                                                      26.9820,  28.0850,  30.9740,  32.0600,  35.4500,  39.9480,
    39.0980,  40.0780,  44.9560,  47.8670,  50.9420,  51.9960,  54.9380,  55.8450,  58.9330,  58.6930,  63.5460,  65.3800,  69.7230,  72.6300,  74.9220,  78.9710,  79.9040,  83.7980,
    85.4680,  87.6200,  88.9060,  91.2240,  92.9060,  95.9500,  97.0000, 101.0700, 102.9100, 106.4200, 107.8700, 112.4100, 114.8200, 118.7100, 121.7600, 127.6000, 126.9000, 131.2900,
    132.9100, 137.3300, 
                        138.9100, 140.1200, 140.9100, 144.2400, 145.0000, 150.3600, 151.9600, 157.2500, 158.9300, 162.5000, 164.9300, 167.2600, 168.9300, 173.0500, 174.9700,
                                178.4900, 180.9500, 183.8400, 186.2100, 190.2300, 192.2200, 195.0800, 196.9700, 200.5900, 204.3800, 207.2000, 208.9800, 209.0000, 210.0000, 222.0000,
    223.0000, 226.0000,
                        227.0000, 232.0400, 231.0400, 238.0300, 237.0000, 244.0000, 243.0000, 247.0000, 247.0000, 251.0000, 252.0000, 257.0000, 258.0000, 259.0000, 266.0000,
                                267.0000, 268.0000, 269.0000, 270.0000, 269.0000, 278.0000, 281.0000, 282.0000, 285.0000, 286.0000, 289.0000, 289.0000, 293.0000, 294.0000, 294.0000
)

# Filled valence numbers
filled_valence_electrons=(2,8,18,32)

# XTB/GFN-2 limitation - Z <= 86
limited_rcov1 = rcov1[0:87]
limited_elements = elements[0:87]
limited_rvdw = rvdw[0:87]
       
lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                    'Ho', 'Er', 'Tm', 'Yb', 'Lu']

limited_lanthanides = [x for x in lanthanides if x in limited_elements]

actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
                    'Es', 'Fm', 'Md', 'No', 'Lr']

limited_actinides = [x for x in actinides if x in limited_elements]

transition_metals = [
            'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co',  'Ni', 'Cu',  'Zn', 
            'Y',  'Zr', 'Nb',  'Mo', 'Tc',  'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
            'Hf',  'Ta', 'W',  'Re',  'Os',  'Ir', 'Pt',  'Au', 'Hg',
            'Rf', 'Db', 'Sg', 'Bh', 'Hs'
            ]

limited_transition_metals = [x for x in transition_metals if x in limited_elements]

post_transition_metals = [ 
            'Al',
            'Ga',
            'In', 'Sn',
            'Tl', 'Pb', 'Bi', 
            'Nh', 'Fl', 'Mc', 'Lv'
        ]

limited_post_transition_metals = [x for x in post_transition_metals if x in limited_elements]

alkali_and_alkaline_earth = [
            'Li', 'Be', 
            'Na', 'Mg', 
            'K', 'Ca', 
            'Rb', 'Sr', 
            'Cs', 'Ba', 
            'Fr', 'Ra'
            ]

alkali_metals = ['Li','Na','K','Rb','Cs','Fr']

alkaline_earth_metals = ['Be','Mg','Ca','Sr','Ba','Ra']

metalloids = ['B','Si','Ge','As','Sb','Te']

limited_alkali_and_alkaline_earth = [x for x in alkali_and_alkaline_earth if x in limited_elements]

heavy_metals = lanthanides + actinides

limited_heavy_metals = limited_lanthanides + limited_actinides

all_metals = lanthanides + actinides + alkali_and_alkaline_earth + post_transition_metals + transition_metals

limited_all_metals = limited_lanthanides + limited_actinides + limited_alkali_and_alkaline_earth \
    + limited_post_transition_metals + limited_transition_metals

# Dictionary for common charges of metals
metal_charge_dict = {
    # Lanthanides
    'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3,
    'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3,
    # Actinides
    'Ac': 3, 'Th': 4, 'Pa': 5, 'U': 4, 'Np': 4, 'Pu': 4, 'Am': 3, 'Cm': 3, 'Bk': 3, 'Cf': 3,
    'Es': 3, 'Fm': 3, 'Md': 3, 'No': 2, 'Lr': 3,
    # First row transition metals 
    'Sc': 3, 'Ti': 4, 'V': 5,  'Cr': 3, 'Mn': 2, 'Fe': 2, 'Co': 2, 'Ni': 2, 'Cu': 2, 'Zn': 2, 
    # Second row transition metals 
    'Y': 3, 'Zr': 4, 'Nb': 5,  'Mo': 6, 'Tc': 5, 'Ru': 2, 'Rh': 1, 'Pd': 2, 'Ag': 1, 'Cd': 2, 
    # Third row transition metals
    'Hf': 4, 'Ta': 5, 'W': 6,  'Re': 4, 'Os': 2, 'Ir': 3, 'Pt': 2, 'Au': 3, 'Hg': 2,
    # 4th row transition metals
    'Rf': 3, 'Db': 5, 'Sg': 6, 'Bh': 7, 'Hs': 8,
    # Post-transition metals
    'Al': 3,
    'Ga': 3,
    'In': 3, 'Sn': 4,
    'Tl': 3, 'Pb': 2, 'Bi': 3, 
    'Nh': 1, 'Fl': 2, 'Mc': 1, 'Lv': 2,
    # Alkali/earth metals
    'Li': 1, 'Be': 2 , 
    'Na': 1, 'Mg': 2, 
    'K': 1, 'Ca': 2, 
    'Rb': 1, 'Sr': 2, 
    'Cs': 1, 'Ba': 2, 
    'Fr': 1, 'Ra': 2
    }

# Dictionary for common spin (alpha-beta) of metals:
metal_spin_dict = {
    # Lanthanides
    'La': 0, 'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 6, 'Gd': 7, 'Tb': 6, 'Dy': 5,
    'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1, 'Lu': 0,
    # Actinides
    'Ac': 0, 'Th': 0, 'Pa': 0, 'U': 2, 'Np': 3, 'Pu': 4, 'Am': 6, 'Cm': 7, 'Bk': 6, 'Cf': 5,
    'Es': 4, 'Fm': 3, 'Md': 2, 'No': 0, 'Lr': 0,
    # First row transition metals
    'Sc': 0, 'Ti': 0, 'V': 0,  'Cr': 3, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1, 'Zn': 0, 
    # Second row transition metals 
    'Y': 0, 'Zr': 0, 'Nb': 0,  'Mo': 0, 'Tc': 2, 'Ru': 4, 'Rh': 2, 'Pd': 0, 'Ag': 0, 'Cd': 0, 
    # Third row transition metals
    'Hf': 0, 'Ta': 0, 'W': 0,  'Re': 1, 'Os': 0, 'Ir': 0, 'Pt': 0, 'Au': 2, 'Hg': 0,
    # 4th row transition metals
    'Rf': 1, 'Db': 0, 'Sg': 0, 'Bh': 0, 'Hs': 0,
    # Post-transition metals
    'Al': 0,
    'Ga': 0,
    'In': 0, 'Sn': 0,
    'Tl': 0, 'Pb': 0, 'Bi': 0, 
    'Nh': 0, 'Fl': 0, 'Mc': 2, 'Lv': 2,
    # Alkali/earth metals
    'Li': 0, 'Be': 0 , 
    'Na': 0, 'Mg': 0, 
    'K': 0, 'Ca': 0, 
    'Rb': 0, 'Sr': 0, 
    'Cs': 0, 'Ba': 0, 
    'Fr': 0, 'Ra': 0
    }

# Dictionary for common spin (alpha-beta) of metals:
second_choice_metal_spin_dict = {
    # Lanthanides
    #'La': 0, 'Ce': 1, -> No real alternative spins
    # Other lanthanides set to 1 lower spin state (nspin-=2)
    'Pr': 0, 'Nd': 1,'Pm': 2,'Sm': 3,'Eu': 4, 'Gd': 5,'Tb': 4, 'Dy': 3, 'Ho': 2, 'Er': 1, 'Tm': 0, 
    #'Yb': 1, 'Lu': 0, -> No real alternative spins
    # Actinides -> Right now actinides are being treated by the lanthanides anyways.
    # 'Ac': 0, 'Th': 0, 'Pa': 0, 'U': 2, 'Np': 2, 'Pu': 4, 'Am': 6, 'Cm': 7, 'Bk': 6, 'Cf': 5,
    # 'Es': 4, 'Fm': 3, 'Md': 2, 'No': 0, 'Lr': 0,
    # First row transition metals
    #'Sc': 0, -> No real alternative spins 
    # Other transition metals set to Low-spin/IS spin state (nspin-=2)
    'Ti': 0, 'V': 1, 'Cr': 0, 'Mn': 1, 'Fe': 0, 'Co': 1, 'Ni': 0, 'Cu': 2, 
    # 'Zn': 0, -> No real alternative spins
    # Second row transition metals 
    # 'Y': 0, -> No real alternative spins
    'Zr': 2, 'Nb': 2,  'Mo': 0, 'Tc': 1, 'Ru': 1, 'Rh': 0, 'Pd': 2, 'Ag': 2, 
    # 'Cd': 0, -> No real alternative spins
    # Third row transition metals
    #'Hf': 0, 'Ta': 0, -> No real alternative spins
    'W': 0,  'Re': 1, 'Os': 2, 'Ir': 2, 'Pt': 0, 'Au': 2, 'Hg': 3,
    # 4th row transition metals
    #'Rf': 1, 'Db': 0, -> No real alternative spins
    'Sg': 2, 'Bh': 2, 'Hs': 2,
    # Post-transition metals
    #'Al': 0,'Ga': 0, 'In': 0, -> No real alternative spins
    'Sn': 2,
    'Tl': 2, 'Pb': 2, 'Bi': 2, 
    'Nh': 2, 'Fl': 2, 'Mc': 0, 'Lv': 0,
    # Alkali/earth metals -> No real alternative spins
    # 'Li': 0, 'Be': 0 , 
    # 'Na': 0, 'Mg': 0, 
    # 'K': 0, 'Ca': 0, 
    # 'Rb': 0, 'Sr': 0, 
    # 'Cs': 0, 'Ba': 0, 
    # 'Fr': 0, 'Ra': 0
    }

metal_CN_dict = {
    # Lanthanides
    'La': [7,8,9], 'Ce': [7,8,9], 'Pr': [7,8,9], 'Nd': [7,8,9], 'Pm': [7,8,9], 'Sm': [7,8,9], 'Eu': [7,8,9], 'Gd': [7,8,9], 'Tb': [7,8,9], 'Dy': [7,8,9],
    'Ho': [7,8,9], 'Er': [7,8,9], 'Tm': [7,8,9], 'Yb': [7,8,9], 'Lu': [7,8,9],
    # Actinides
    'Ac': [6,7,8,9], 'Th': [6,7,8,9], 'Pa': [6,7,8,9], 'U': [6,7,8,9], 'Np': [6,7,8,9], 'Pu': [6,7,8,9], 'Am': [6,7,8,9], 'Cm': [6,7,8,9], 'Bk': [6,7,8,9], 'Cf': [6,7,8,9],
    'Es': [6,7,8,9], 'Fm': [6,7,8,9], 'Md': [6,7,8,9], 'No': [6,7,8,9], 'Lr': [6,7,8,9],
    # First row transition metals 
    'Sc': [4,6], 'Ti': [4,6], 'V': [4,6],  'Cr': [4,6], 'Mn': [4,6], 'Fe': [4,6], 'Co': [4,6], 'Ni': [4,6], 'Cu': [4,6], 'Zn': [4,6], 
    # Second row transition metals 
    'Y': [4,6], 'Zr': [4,6], 'Nb': [4,6],  'Mo': [4,6], 'Tc': [4,6], 'Ru': [4,6], 'Rh': [4,6], 'Pd': [4,6], 'Ag': [4,6], 'Cd': [4,6], 
    # Third row transition metals
    'Hf': [4,6], 'Ta': [4,6], 'W': [4,6],  'Re': [4,6], 'Os': [4,6], 'Ir': [4,6], 'Pt': [4,6], 'Au': [4,6], 'Hg': [4,6],
    # 4th row transition metals
    'Rf': [4,6], 'Db': [4,6], 'Sg': [4,6], 'Bh': [4,6], 'Hs': [4,6],
    # Post-transition metals
    'Al': [4,6],
    'Ga': [4,6],
    'In': [4,6], 'Sn': [4,6],
    'Tl': [4,6], 'Pb': [4,6], 'Bi': [4,6], 
    'Nh': [4,6], 'Fl': [4,6], 'Mc': [4,6], 'Lv': [4,6],
    # Alkali/earth metals
    'Li': [4,6,8], 'Be': [4,6,8], 
    'Na': [4,6,8], 'Mg': [4,6,8], 
    'K': [4,6,8], 'Ca': [4,6,8], 
    'Rb': [4,6,8], 'Sr': [4,6,8], 
    'Cs': [4,6,8], 'Ba': [4,6,8], 
    'Fr': [4,6,8], 'Ra': [4,6,8]
    }

# GFN2-xTB single atom energies see io_xtb_calc.calc_xtb_ref_dict for reference.
xtb_single_atom_ref_es = {'H': -10.707211383396714, 
 'He': -47.432891698445495,
 'Li': -4.900000175455953,
 'Be': -15.486162554518229,
 'B': -25.917120371751917,
 'C': -48.847445262804705,
 'N': -71.00681805517411,
 'O': -102.57117256025786,
 'F': -125.69864294466228,
 'Ne': -161.42379378015602,
 'Na': -4.5469341628136,
 'Mg': -12.67981645403045,
 'Al': -24.63524632585141,
 'Si': -42.76062738849161,
 'P': -64.70342656532247,
 'S': -85.66881795502964,
 'Cl': -121.97572181135438,
 'Ar': -116.4386981693596,
 'K': -4.510348161503552,
 'Ca': -10.113012362120033,
 'Sc': -23.243511328068923,
 'Ti': -37.19952407328606,
 'V': -46.75256780519609,
 'Cr': -47.551829431041725,
 'Mn': -71.27032275557823,
 'Fe': -80.14401872919589,
 'Co': -95.42461229338535,
 'Ni': -127.03165993183204,
 'Cu': -101.98844165193441,
 'Zn': -14.354588513999577,
 'Ga': -29.418551497128806,
 'Ge': -49.249990531246375,
 'As': -60.93788396017277,
 'Se': -84.9113939279083,
 'Br': -110.16092538829801,
 'Kr': -116.24311016235609,
 'Rb': -4.353793155897734,
 'Sr': -12.583384450577476,
 'Y': -32.513603426171194,
 'Zr': -35.664705344038914,
 'Nb': -48.468943633382366,
 'Mo': -47.103307394758325,
 'Tc': -67.47438866812136,
 'Ru': -77.53081994875949,
 'Rh': -106.01053867238016,
 'Pd': -119.99800276414216,
 'Ag': -103.99479372376777,
 'Cd': -14.504682519374043,
 'In': -30.63832754080574,
 'Sn': -54.773706848758266,
 'Sb': -58.891664996211276,
 'Te': -81.88153581941742,
 'I': -102.84897812647684,
 'Xe': -105.67782578404142,
 'Cs': -4.04170614472273,
 'Ba': -11.800000422526582,
 'La': -32.78408804872581,
 'Ce': -24.476883526857367,
 'Pr': -24.30981661010049,
 'Nd': -24.142748696083178,
 'Pm': -23.975681784199946,
 'Sm': -23.80861387345064,
 'Eu': -23.641546957707547,
 'Gd': -23.474481043951293,
 'Tb': -23.30741258426,
 'Dy': -23.140346671193583,
 'Ho': -22.973279758167216,
 'Er': -22.80621184515164,
 'Tm': -22.63914493213882,
 'Yb': -22.472076019126465,
 'Lu': -22.305010106114292,
 'Hf': -35.70130487846373,
 'Ta': -51.83261682874072,
 'W': -60.25302489724785,
 'Re': -81.80817630252322,
 'Os': -81.31693407962229,
 'Ir': -17.446427065227073,
 'Pt': -120.7493981742078,
 'Au': -103.47454570514782,
 'Hg': -23.076132826294845,
 'Tl': -39.14861684553654,
 'Pb': -59.99677903578255,
 'Bi': -61.67878109601047,
 'Po': -74.41720155213928,
 'At': -81.64893636735967,
 'Rn': -104.97843975899828}
    
functional_groups_dict = {
'methyl':'C',
'ethyl':'CC',
'phenyl':'c1ccccc1',
'bromo':'Br',
'iodo':'I',
'chloro':'Cl',
'amino':'N',
'hydroxyl':'O',
'thiol':'S',
'carbonyl':'C#[O+]',
'cyano':'C#N',
'fluoro':'F',
'trichloro':'C(Cl)(Cl)Cl',
'trifluro':'C(F)(F)F',
'tribromo':'C(Br)(Br)Br',
'ether':'OC',
'carboxyl':'C(=O)O',
'carboxylate':'C(=O)[O-]',
'ester':'C(=O)OC',
'ketone':'C(=O)C',
'aldehyde':'C(=O)',
'amide':'C(=O)N(C)C',
'cyanimide':'N(C)N',
'phosphonate':'[PH+]([O-])(O)O',
'2-hydroxypyradine':'c1nc(ccc1)O',
'2-methylbenzoic_acid':'c1cc(ccc1C(=O)O)C'
}

solvents_dict = {
'acetone':'CC(=O)C',
'acetonitrile':'CC#N',
'aniline':'C1=CC=C(C=C1)N',
'benzaldehyde':'C1=CC=C(C=C1)C=O',
'benzene':'C1=CC=CC=C1 ',
'CH2Cl2':'C(Cl)Cl',
'CHCl3':'C(Cl)(Cl)Cl',
'CS2':'S=C=S',
'dioxane':'O1CCOCC1',
'dmf':'CN(C)C=O',
'dmso':'CS(=O)C',
'ether':'CCOCC',
'ethylacetate':'CCOC(=O)C',
'furane':'C1=COC=C1',
'hexandecane':'CCCCCCCCCCCCCCCC',
'hexane':'CCCCCC',
'methanol':'CO',
'nitromethane':'C[N+](=O)[O-]',
'octanol':'CCCCCCCCO', 
'phenol':'Oc1ccccc1',
'toluene':'CC1=CC=CC=C1',
'thf':'C1CCCO1',
'water':'O'
}

# Commonly used ligands dictionary largely from : https://en.wikipedia.org/wiki/Ligand (accessed 5/24/2022)
ligands_dict = {
'water':{'smiles':'O','coordList':[0],'ligType':'mono'},
'hydroxyl':{'smiles':'[OH-]','coordList':[0],'ligType':'mono'},
'oxo':{'smiles':'[O-2]','coordList':[0],'ligType':'mono'},
'hydride':{'smiles':'[H-]','coordList':[0],'ligType':'mono'},
'sulfide':{'smiles':'[S-2]','coordList':[0],'ligType':'mono'},
'pyradine':{'smiles':'c1ccncc1','coordList':[3],'ligType':'mono'},
'bipyradine':{'smiles':'n1ccccc1-c2ccccn2','coordList':[0,11],'ligType':'bi_cis'},
'bipy':{'smiles':'n1ccccc1-c2ccccn2','coordList':[0,11],'ligType':'bi_cis'},
'terpy':{'smiles':'c1ccnc(c1)c2cccc(n2)c3ccccn3','coordList':[3,11,17],'ligType':'tri_mer'},
'terpyradine':{'smiles':'c1ccnc(c1)c2cccc(n2)c3ccccn3','coordList':[3,11,17],'ligType':'tri_mer'},
'fluoride':{'smiles':'[F-]','coordList':[0],'ligType':'mono'},
'chloride':{'smiles':'[Cl-]','coordList':[0],'ligType':'mono'},
'bromide':{'smiles':'[Br-]','coordList':[0],'ligType':'mono'},
'iodide':{'smiles':'[I-]','coordList':[0],'ligType':'mono'},
'acac':{'smiles':'CC(=CC(=O)C)[O-]','coordList':[4,6],'ligType':'bi_cis'},
'methanol':{'smiles':'CO','coordList':[1],'ligType':'mono'},
'methoxy':{'smiles':'C[O-]','coordList':[1],'ligType':'mono'},
'ethanol':{'smiles':'CCO','coordList':[2],'ligType':'mono'},
'ethoxy':{'smiles':'CC[O-]','coordList':[2],'ligType':'mono'},
'octanol':{'smiles':'CCCCCCCCO','coordList':[8],'ligType':'mono'},
'nitrate_bi':{'smiles':'[N+](=O)([O-])[O-]','coordList':[2,3],'ligType':'bi_cis_chelating'},
'nitrate_mono':{'smiles':'[N+](=O)([O-])[O-]','coordList':[2],'ligType':'mono'},
'thiocyanite':{'smiles':'[S-]C#N','coordList':[0],'ligType':'mono'},
'isothiocyanite':{'smiles':'S=C=[N-]','coordList':[2],'ligType':'mono'},
'azide':{'smiles':'[N-]=[N+]=[N-]','coordList':[0],'ligType':'mono'},
'benzoate':{'smiles':'C1=CC=C(C=C1)C(=O)[O-]','coordList':[7,8],'ligType':'bi_cis_chelating'}, 
'benzoate_mono':{'smiles':'C1=CC=C(C=C1)C(=O)[O-]','coordList':[8],'ligType':'mono'},
'benzoic_acid':{'smiles':'C1=CC=C(C=C1)C(=O)O','coordList':[7],'ligType':'mono'},
'oxalate':{'smiles':'C(=O)(C(=O)[O-])[O-]','coordList':[4,5],'ligType':'bi_cis'},
'nitrite_o':{'smiles':'N(=O)[O-]','coordList':[2],'ligType':'mono'},
'nitrite_n':{'smiles':'N(=O)[O-]','coordList':[0],'ligType':'mono'},
'acetonitrile':{'smiles':'CC#N','coordList':[2],'ligType':'mono'},
'ammonia':{'smiles':'N','coordList':[0],'ligType':'mono'},
'thf':{'smiles':'C1CCCO1','coordList':[4],'ligType':'mono'},
'en':{'smiles':'NCCN','coordList':[0,3],'ligType':'bi_cis'},
'ethylenediamine':{'smiles':'NCCN','coordList':[0,3],'ligType':'bi_cis'},
'phen':{'smiles':'c1cc2ccc3cccnc3c2nc1','coordList':[9,12],'ligType':'bi_cis'},
'phenanthroline':{'smiles':'c1cc2ccc3cccnc3c2nc1','coordList':[9,12],'ligType':'bi_cis'},
'tpp':{'smiles':'c1ccccc1P(c2ccccc2)c3ccccc3','coordList':[6],'ligType':'mono'},
'triphenylphosphine':{'smiles':'c1ccccc1P(c2ccccc2)c3ccccc3','coordList':[6],'ligType':'mono'},
'cyanide':{'smiles':'[C-]#N','coordList':[0],'ligType':'mono'},
'isocyanide':{'smiles':'[C-]#N','coordList':[1],'ligType':'mono'},
'carbonyl':{'smiles':'[C-]#[O+]','coordList':[0],'ligType':'mono'},
'dppe':{'smiles':'P(c1ccccc1)(c2ccccc2)CCP(c3ccccc3)c4ccccc4','coordList':[0,15],'ligType':'bi_cis'},
'bisdiphenylphosphinoethane':{'smiles':'P(c1ccccc1)(c2ccccc2)CCP(c3ccccc3)c4ccccc4','coordList':[0,15],'ligType':'bi_cis'},
'dppm':{'smiles':'P(c1ccccc1)(c2ccccc2)CP(c3ccccc3)c4ccccc4','coordList':[0,14],'ligType':'bi_cis'},
'bisdiphenylphosphinomethane':{'smiles':'P(c1ccccc1)(c2ccccc2)CP(c3ccccc3)c4ccccc4','coordList':[0,14],'ligType':'bi_cis'},
'corrole':{'smiles':'C1=CC2=CC3=CC=C([N-]3)C4=NC(=CC5=CC=C([N-]5)C=C1[N-]2)C=C4','coordList':[8,10,17,20],'ligType':'tetra_planar'},
'9-crown-3':{'smiles':'C1COCCOCCO1','coordList':[2,5,8],'ligType':'tri_fac'},
'12-crown-4':{'smiles':'O1CCOCCOCCOCC1', 'coordList':[0,3,6,9],'ligType':'tetra_pyramidal'},
'15-crown-5':{'smiles':'C1COCCOCCOCCOCCO1','coordList':[2,5,8,11,14],'ligType':'penta_planar'},
'18-crown-6':{'smiles':'O1CCOCCOCCOCCOCCOCC1','coordList':[0,3,6,9,12,15],'ligType':'hexa_planar'},
'222-cryptand':{'smiles':'C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2','coordList':[2,5,8,11,14,17,20,23],
              'ligType':'octa_trigonal_prismatic_triangle_face_bicapped'},
'cp':{'smiles':'C1=C[CH-]C=C1','coordList':[0,1,2,3,4],'ligType':'sandwich'},
'cyclopentadienyl':{'smiles':'C1=C[CH-]C=C1','coordList':[0,1,2,3,4],'ligType':'sandwich'},
'benzene':{'smiles':'c1ccccc1','coordList':[0,1,2,3,4,5],'ligType':'sandwich'},
'cp_m':{'smiles':'C[C-]1C(=C(C(=C1C)C)C)C','coordList':[1,2,3,4,5],'ligType':'sandwich'},
'pentamethylcyclopentadienyl':{'smiles':'C[C-]1C(=C(C(=C1C)C)C)C','coordList':[1,2,3,4,5],'ligType':'sandwich'},
'dien':{'smiles':'NCCNCCN','coordList':[0,3,6],'ligType':'tri_fac'},
'diethylenetriamine':{'smiles':'NCCNCCN','coordList':[0,3,6],'ligType':'tri_fac'},
'dmgh-':{'smiles':'C\C(=N\O)\C(\C)=N\[O-]','coordList':[2,6],'ligType':'bi_cis'},
'dota':{'smiles':'O=C(O)CN(CC1)CCN(CC(=O)O)CCN(CC(=O)O)CCN1CC(=O)O','coordList':[0,4,9,12,16,23,19,26],
        'ligType':'octa_square_antiprismatic'}, 
'tetraxetan':{'smiles':'O=C(O)CN(CC1)CCN(CC(=O)O)CCN(CC(=O)O)CCN1CC(=O)O','coordList':[0,4,9,12,16,23,19,26],
        'ligType':'octa_square_antiprismatic'},
'dtpa':{'smiles':'C(CN(CC(=O)O)CC(=O)O)N(CCN(CC(=O)O)CC(=O)O)CC(=O)O', 
        'coordList':[2,5,9,11,14,17,21,25],'ligType':'octa_square_antiprismatic'},
'3,4,3-li(1-2-hopo)':{'smiles':'C1=CC(=O)N(C(=C1)C(=O)NCCCN(CCCCN(CCCNC(=O)C2=CC=CC(=O)N2[O-])C(=O)C3=CC=CC(=O)N3[O-])C(=O)C4=CC=CC(=O)N4[O-])[O-]', 
        'coordList':[3,53,52,50,30,32,40,42],'ligType':'octa_square_antiprismatic'},
'dheba':{'smiles':'CCCCC(CC)COP(=O)([O-])OCC(CC)CCCC','coordList':[10,11],'ligType':'bi_cis_bulky'},
'edta':{'smiles':'[O-]C(=O)CN(CCN(CC([O-])=O)CC([O-])=O)CC([O-])=O','coordList':[0,4,7,10,14,18],
        'ligType':'hexa_octahedral'},
'ethylenediaminetriacetate':{'smiles':'C(C[NH+](CC(=O)[O-])CC(=O)[O-])NCC(=O)[O-]',
        'coordList':[2,6,10,11,15],'ligType':'penta_square_pyramidal'},
'egta':{'smiles':'O=C([O-])CN(CC(=O)[O-])CCOCCOCCN(CC(=O)[O-])CC(=O)[O-]',
        'coordList':[2,4,8,11,14,17,21,25],'ligType':'octa_trigonal_prismatic_triangle_face_bicapped'},
'glycine':{'smiles':'C(C(=O)[O-])N','coordList':[3,4],'ligType':'bi_cis'},
'porphyrin':{'smiles':'C1=CC2=CC3=CC=C([N-]3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C1[N-]2','coordList':[8,14,20,23],
             'ligType':'tetra_planar'},
'corrin':{'smiles':'C1CC2=NC1C3CCC(=N3)C=C4CCC(=N4)C=C5CCC(=C2)[N-]5','coordList':[3,9,15,22],
          'ligType':'tetra_planar'},
'ida':{'smiles':'[O-]C(=O)CNCC([O-])=O','coordList':[0,4,7],'ligType':'tri_mer'},
'iminodiacetate':{'smiles':'[O-]C(=O)CNCC([O-])=O','coordList':[0,4,7],'ligType':'tri_mer'},
'nitrosyl':{'smiles':'[N+]=O','coordList':[0],'ligType':'mono'},
'nta':{'smiles':'O=C([O-])CN(CC(=O)[O-])CC(=O)[O-]','coordList':[2,4,7,12],'ligType':'tetra_seesaw'},
'pyrazine':{'smiles':'c1cnccn1','coordList':[2],'ligType':'mono'},
'scorpionate':{'smiles':'[CH-]1C=C[N+](=C(C(=O)[O-])N2C=CC=N2)[N-]1','coordList':[7,12,13],
               'ligType':'tri_fac'},
'tp':{'smiles':'[BH-](N1C=CC=N1)(N2C=CC=N2)N3C=CC=N3','coordList':[5,10,15],'ligType':'tri_fac'},
'trithia-9-crown-3':{'smiles':'C1CSCCSCCS1','coordList':[2,5,8],'ligType':'tri_fac'},
'sulfite':{'smiles':'[O-]S(=O)[O-]','coordList':[1],'ligType':'mono'},
'isosulfite':{'smiles':'[O-]S(=O)[O-]','coordList':[0],'ligType':'mono'},
'tacn':{'smiles':'C1CNCCNCCN1','coordList':[2,5,8],'ligType':'tri_fac'},
'tricyclhexylphosphine':{'smiles':'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3','coordList':[6],'ligType':'mono'},
'trien':{'smiles':'NCCNCCNCCN','coordList':[0,3,6,9], 'ligType':'tetra_trigonal_pyramidal'},
'tmp':{'smiles':'CP(C)C','coordList':[1],'ligType':'mono'},
'triso-tolylphosphine':{'smiles':'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C','coordList':[7],'ligType':'mono'},
'tren':{'smiles':'NCCN(CCN)CCN','coordList':[0,3,6,9],'ligType':'tetra_trigonal_pyramidal'},
'tris2-diphenylphosphinoethylamine':{'smiles':'c1ccc(cc1)P(CCN(CCP(c2ccccc2)c3ccccc3)CCP(c4ccccc4)c5ccccc5)c6ccccc6',
                                     'coordList':[6,9,12,27], 'ligType':'tetra_trigonal_pyramidal'},
# 'trpoylium':{'smiles':'c1=cc=c[cH+]c=c1','coordList':[0]} # Can't get smiles to work with obabel/mmff94
'amine':{'smiles':'N','coordList':[0],'ligType':'mono'},
'co2':{'smiles':'O=C=O','coordList':[1],'ligType':'mono'},
'phosphorus_trifluoride':{'smiles':'FP(F)F','coordList':[1],'ligType':'mono'},
'topo':{'smiles':'CCCCCCCCP(=O)(CCCCCCCC)CCCCCCCC','coordList':[9],'ligType':'mono'},
'trimethyl_phosphite':{'smiles':'COP(OC)OC','coordList':[2],'ligType':'mono'},
'trimethyl_phosphine_oxide':{'smiles':'CP(=O)(C)C','coordList':[1],'ligType':'mono'}, 
'cyanex-301':{'smiles':'CC(CC(C)(C)C)CP(=S)(CC(C)CC(C)(C)C)[S-]','coordList':[9,18],'ligType':'bi_cis_chelating'},
'bmptt':{'smiles':'CC1=NN(C(=S)[C@H]1C(=O)c1ccccc1)c1ccccc1','coordList':[5,8],'ligType':'bi_cis'},
'dpphen':{'smiles':'C1=CC=C(C=C1)C2=C3C=CC4=C(C=CN=C4C3=NC=C2)C5=CC=CC=C5','coordList':[14,17],'ligType':'bi_cis'},
'bis_4-chlorophenyl_dithiophosphinate':{'smiles':'C1=CC(=CC=C1P(=S)(C2=CC=C(C=C2)Cl)[S-])Cl',
                                        'coordList':[7,15],'ligType':'bi_cis'},
# 'talspeak-hdehdga':{'smiles':'CCCCC(CC)CC(CC(CC)CCCC)(C(=O)O)OCC(=O)N'} # Multiprotic
'btp':{'smiles':'c1cnnc(n1)c1cccc(n1)c1nccnn1','coordList':[3,11,17],'ligType':'tri_mer'},
'kryptofix-22':{'smiles':'C1COCCOCCN(CCOCCOCCN1C)C','coordList':[2,5,8,11,14,17],'ligType':'hexa_planar'}, # Functionalize at 18,19
'kryptofix-222':{'smiles':'C1COCCOCCN2CCOCCOCCN1CCOCCOCC2','coordList':[2,5,8,11,14,17,20,23], #
                    'ligType':'octa_trigonal_prismatic_triangle_face_bicapped'},
'kryptofix-211':{'smiles':'C1COCCN2CCOCCN1CCOCCOCC2','coordList':[2,5,8,11,14,17], #
                    'ligType':'hexa_trigonal_prismatic'},
'oxalacetic_acid':{'smiles':'C(C(=O)C(=O)O)C(=O)O','coordList':[2,4,7],'ligType':'tri_fac'}, #
'oxalacetic_acid_bi':{'smiles':'C(C(=O)C(=O)O)C(=O)O','coordList':[2,4],'ligType':'bi_cis'},
'oxalacetate':{'smiles':'C(C(=O)C(=O)[O-])C(=O)O','coordList':[2,5,7],'ligType':'tri_fac'}, #
'oxalacetate_bi':{'smiles':'C(C(=O)C(=O)[O-])C(=O)O','coordList':[2,5,7],'ligType':'bi_cis'},
'oxalbiacetate':{'smiles':'C(C(=O)C(=O)[O-])C(=O)[O-]','coordList':[2,5,8],'ligType':'tri_fac'},
'oxalbiacetate_bi':{'smiles':'C(C(=O)C(=O)[O-])C(=O)[O-]','coordList':[2,5],'ligType':'bi_cis'},
'valine':{'smiles':'CC(C)C(C(=O)O)N','coordList':[5,6],'ligType':'bi_cis'}, #
'valinate_bi_n':{'smiles':'CC(C)C(C(=O)[O-])N','coordList':[6,7],'ligType':'bi_cis'},
'valinate_bi_o':{'smiles':'CC(C)C(C(=O)[O-])N','coordList':[5,7],'ligType':'bi_cis_chelating'},
'oxime':{'smiles':'c1(O)c(C(C)(C)C)cc(C(C)(C)C)cc1(C(=NO))','coordList':[1,16],'ligType':'bi_cis'},
'pyridine_ester':{'smiles':'COC(=O)C1=CC(=CN=C1)C(=O)OC','coordList':[8],'ligType':'mono'},
'tedga':{'smiles':'CCN(CC)C(=O)COCC(=O)N(CC)CC','coordList':[6,8,11],'ligType':'tri_mer_bent'},
'hydroxyquinoline':{'smiles':'C1=CC2=C(C(=C(C)1)O)N=CC=C2','coordList':[7,8],'ligType':'bi_cis'}, # Functionalize 6
'bi_benzimidazole':{'smiles':'C1=CC=C2C(=C1)NC(=N2)C3=NC4=CC=CC=C4N3','coordList':[8,10],'ligType':'bi_cis'} # Functionalize 6,17 with esters
}


def convert_actinides_lanthanides(elem,inverse=False):
    """convert_actinides_lanthanides [summary]

    Parameters
    ----------
    elem : str
        element (metal) to consider
    inverse : bool, optional
        wheter to re-convert lanthanides to actinides in final structure, by default False

    Returns
    -------
    out : str
        new metal symbol
    isact : bool
        whether the elem passed was an actinide or is an actinide complex
    """
    acts = actinides
    lns = lanthanides
    isact = False
    if elem in acts:
        out = lns[acts.index(elem)]
        isact = True
    elif inverse:
        out = acts[lns.index(elem)]
        isact = True
    else:
        out = elem
    return out, isact

def larger_map_metal(inputDict):
    """larger_map_metal 
    Map metal to "similar" metal with covalent radii about 1.3 times the normal one

    Parameters
    ----------
    inputDict : dict
        input dictionary
    
    Returns
    ----------
    newinpDict : dict
        new input dictionary
    match : bool
        whether or not a larger element exists to try
    metal : str 
        what the original string was
    """
    metal = inputDict['core']['smiles'].strip('[').strip(']')
    metal, isact = convert_actinides_lanthanides(metal) # Ensure lanthanide used as reference for size not actinide
    covrad_metal = rcov1[elements.index(metal)]
    if metal in transition_metals:
        rcov_tms = np.array([limited_rcov1[limited_elements.index(x)] for x in limited_transition_metals])
        matches = np.isclose(rcov_tms,1.3*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_transition_metals[np.where(matches)[0][0]]
        else:
            match = None
    elif metal in post_transition_metals:
        rcov_tms = [limited_rcov1[limited_elements.index(x)] for x in limited_post_transition_metals]
        matches = np.isclose(rcov_tms,1.3*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_post_transition_metals[np.where(matches)[0][0]]
        else:
            match = None
    elif metal in alkali_and_alkaline_earth:
        rcov_tms = [limited_rcov1[limited_elements.index(x)] for x in limited_alkali_and_alkaline_earth]
        matches = np.isclose(rcov_tms,1.3*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_alkali_and_alkaline_earth[np.where(matches)[0][0]]
        else:
            match = None
    elif metal in heavy_metals:
        rcov_tms = [limited_rcov1[limited_elements.index(x)] for x in limited_heavy_metals]
        matches = np.isclose(rcov_tms,1.3*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_heavy_metals[np.where(matches)[0][0]]
        else:
            match = None
    if (not match): # Try to see if any metals within tolerance
        rcov_tms = [limited_rcov1[limited_elements.index(x)] for x in limited_all_metals]
        matches = np.isclose(rcov_tms,1.3*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_all_metals[np.where(matches)[0][0]]
        else:
            matches = np.isclose(rcov_tms,1.3*covrad_metal,atol=1e-1)
            if np.any(matches): # Try again with looser tolerance
                match = limited_all_metals[np.where(matches)[0][0]]
            else:
                match = None
    if isinstance(match,str):
        newinpDict = copy.deepcopy(inputDict)
        newinpDict['core']['smiles'] = '['+match+']'
        newinpDict['core']['metal'] = match
        match = True
    else:
        match = False
        newinpDict = copy.deepcopy(inputDict)
    return newinpDict, match, metal

def smaller_map_metal(inputDict):
    """larger_map_metal 
    Map metal to "similar" metal with covalent radii about 0.8 times the normal one

    Parameters
    ----------
    inputDict : dict
        input dictionary
    
    Returns
    ----------
    newinpDict : dict
        new input dictionary
    match : bool
        whether or not a larger element exists to try
    metal : str 
        what the original string was
    """
    metal = inputDict['core']['smiles'].strip('[').strip(']')
    metal, isact = convert_actinides_lanthanides(metal)
    covrad_metal = rcov1[elements.index(metal)]
    if metal in transition_metals:
        rcov_tms = np.array([limited_rcov1[limited_elements.index(x)] for x in limited_transition_metals])
        matches = np.isclose(rcov_tms,0.8*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_transition_metals[np.where(matches)[0][0]]
        else:
            match = None
    elif metal in post_transition_metals:
        rcov_tms = [limited_rcov1[limited_elements.index(x)] for x in limited_post_transition_metals]
        matches = np.isclose(rcov_tms,0.8*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_post_transition_metals[np.where(matches)[0][0]]
        else:
            match = None
    elif metal in alkali_and_alkaline_earth:
        rcov_tms = [limited_rcov1[limited_elements.index(x)] for x in limited_alkali_and_alkaline_earth]
        matches = np.isclose(rcov_tms,0.8*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_alkali_and_alkaline_earth[np.where(matches)[0][0]]
        else:
            match = None
    elif metal in heavy_metals:
        rcov_tms = [limited_rcov1[elements.index(x)] for x in limited_heavy_metals]
        matches = np.isclose(rcov_tms,0.8*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_heavy_metals[np.where(matches)[0][0]]
        else:
            match = None
    if (not match): # Try to see if any metals within tolerance
        rcov_tms = [limited_rcov1[limited_elements.index(x)] for x in limited_all_metals]
        matches = np.isclose(rcov_tms,0.8*covrad_metal,atol=5e-2)
        if np.any(matches):
            match = limited_all_metals[np.where(matches)[0][0]]
        else:
            matches = np.isclose(rcov_tms,0.8*covrad_metal,atol=1e-1)
            if np.any(matches): # Try again with looser tolerance
                match = limited_all_metals[np.where(matches)[0][0]]
            else:
                match = None
    if isinstance(match,str):
        newinpDict = copy.deepcopy(inputDict)
        newinpDict['core']['smiles'] = '['+match+']'
        newinpDict['core']['metal'] = match
        match = True
    else:
        match = False
        newinpDict = copy.deepcopy(inputDict)
    return newinpDict, match, metal

def map_metal_radii(inputDict,larger=False,
                    larger_factor=1.3,smaller_factor=0.8):
    """map_metal_radii
    Rescale metal covalent radii and vdwrad to about 1.3 times the normal one

    Parameters
    ----------
    inputDict : dict
        input dictionary
    larger : bool
        wheter to make the radii larger or smaller, default is False
    larger_factor : float
        scale by which to make the radii larger, default is 1.3
    smaller_factor : float
        scale by which to make the radii smaler, default is 0.8

    
    Returns
    ----------
    newinpDict : dict
        new input dictionary
    match : bool
        whether or not a larger element exists to try
    metal : str 
        what the original string was 
    """
    metal = inputDict['core']['smiles'].strip('[').strip(']')
    covrad_metal = rcov1[elements.index(metal)]
    vdwrad_metal = rvdw[elements.index(metal)]
    newinpDict = copy.deepcopy(inputDict)
    if larger:
        newinpDict['parameters']['covrad_metal'] = covrad_metal*larger_factor
        newinpDict['parameters']['rvdw_metal'] = vdwrad_metal*larger_factor
        newinpDict['parameters']['scaled_radii_factor'] = larger_factor
    else:
        newinpDict['parameters']['covrad_metal'] = covrad_metal*smaller_factor
        newinpDict['parameters']['rvdw_metal'] = vdwrad_metal*smaller_factor
        newinpDict['parameters']['scaled_radii_factor'] = smaller_factor
    return newinpDict