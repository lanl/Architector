# Architector

Architector is a 3D chemical structure generation software package designed to take minimal 2D information about ligands and metal centers and generates chemically sensible 3D conformers and stereochemistry of the organometallic compounds.
It is capable of high-throughput in-silico construction of s-, p-, d-, and f-block organometallic complexes. Architector represents a transformative step towards cross-periodic table computational design of metal complex chemistry.

## Installation

Conda installation recommended. The conda-forge distribution can be installed via: 

```bash
conda install -c conda-forge architector
```

* In case a developer version of the sofware is required in the root directory for Architector run:

```bash
conda env create -f environment.yml
conda activate architector
pip install -e .
```

## Useful Tools/Examples:

1. See tutorials for basic introduction to capabilties and code examples: `documentation/tutorials/`
2. Reference for core and ligand geometry labels see: `documentation/view_default_core_ligand_types.ipynb`
3. Utility for aiding in determining ligand coordination sites see: `utils/ligand_viewing_coordinating_atom_selecting.ipynb`

* Note that ligands used in (3) can even be drawn in [Avogadro](https://avogadro.cc/) and copied as SMILES strings into this analysis.
* If other analyses are used to determine the coordinating atom indices we can't guarantee the generated structure will match what was input. If generating complexes with new ligands we HIGHLY recommend using the utility in (3)

## XTB (backend) Potentially Useful References:
* [Available Solvents](https://xtb-docs.readthedocs.io/en/latest/gbsa.html)
* [Available Methods](https://xtb-python.readthedocs.io/en/latest/general-api.html)
* [ASE Calculator](https://xtb-python.readthedocs.io/en/latest/ase-calculator.html)
* [XTB Documentation](https://xtb-docs.readthedocs.io/en/latest/contents.html)

## Basic Use of complex construction functionality:

```python
from architector import build_complex
out = build_complex(inputDict)
```

## Input dictionary structure and recommendations:
```python
inputDict = {
################ Core (metal) structure and optional definitions #####################
# Requires input for what metal and what type of coordination environments to sample #

"core": {
    "metal":'Fe', 
    # "coordList" OR "coreType" OR "coreCN" (Suggested!)
    'coordList': None, 
    # Handles user-defined list of core coordination vectors e.g.
    # [
    #     [2., 0.0, 0.0],
    #     [0.0, 2., 0.0],
    #     [0.0, 0.0, 2.],
    #     [-2., 0.0, 0.0],
    #     [0.0, -2., 0.0],
    #     [0.0, 0.0, -2.] 
    # ] -> gets defined as 'user_geometry'
    "coreType": None, 
    # e.g. 'octahedral' ....
    # or list of coreTypes - e.g. ['octahedral','trigonal_prismatic','tetrahedral']
    "coreCN": 6 #(SUGGETED!)
    # Core coordination number (CN) (int)
    # Will calculate all possible geometries with the given coreCN 
    # Tends to sample the metal space better than other options.
    # OR list of CNs [4,6] -> Will calculate all possible geometries with these CNs.
    # NOTE that if nothing is passed, a list of common coreCNs will be used to attempt structure generation.
    }, 
############## Ligands  list and optional definitions ####################
# Requires either smiles and metal-coordinating site definitions or default ligand names  #

"ligands": [
    {"smiles":"n1ccccc1-c2ccccn2",
    # Smiles required. Can also be generated and drawn using avogadro molecular editor.
    "coordList":[0, 11], 
    # Coordination sites corresponding to the SMILES atom connecting to the metal
    # Can be determined/assigned manually using utils/ligand_viewing_coordinating_atom_selecting.ipynb
    # Alternatively [[0,1],[11,2]], In this case it forces it to be map to the user-defined core coordinating sites.
    'ligType':'bi_cis'
    # Optional, but desirable - if not-specified will will assign the best ligType guess using a brute force assignment that can be slow. 
    }, 
    ],
    # NOTE - multiple ligands should be added to fill out structure if desired.

############## Additional Parameters for the structural generation  ####################
# Here, metal oxdiation state and spin state, methods for evaluating complexes during construction, #
# And many other options are defined, but are often handled automatically by Architector in the background #

"parameters" = {
    ######## Electronic parameters #########
    "metal_ox": None, # Oxidation State
    "metal_spin": None, # Spin State
    "full_spin": None, # Assign spin to the full complex (overrides metal_spin)
    "full_charge": None, # Assign charge to the complex (overrides ligand charges and metal_ox)!
        
    # Method parameters.
    "full_method": "GFN2-xTB", # Which  method to use for final cleaning/evaulating conformers. 
    "assemble_method": "GFN2-xTB", # Which method to use for assembling conformers. 
    # For very large speedup - use "GFN-FF", though this is much less stable (especially for Lanthanides)
    # Additionaly, it is possible to use "UFF" - which is extremely fast. Though it is recommend to perform an XTB-level optimization
    # for the "full_method", or turn "relaxation" off.
    "xtb_solvent": 'none', # Add any named XTB solvent!
    "xtb_accuracy": 1.0, # Numerical Accuracy for XTB calculations
    "xtb_electronic_temperature": 300, # In K -> fermi smearing - increase for convergence on harder systems
    "xtb_max_iterations": 250, # Max iterations for xtb SCF.
    "force_generation":False, # Whether to force the construction to proceed without xtb energies - defaults to UFF evaluation
    # in cases of XTB outright failure. Will still enforce sanity checks on output structures.

    # Covalent radii and vdw radii of the metal if nonstandard radii requested.
    "vdwrad_metal": vdwrad_metal,
    "covrad_metal": covrad_metal,

    ####### Conformer parameters and information stored ########
    "n_conformers": 1, # Number of metal-core symmetries at each core to save / relax
    "return_only_1": False, # Only return single relaxed conformer (do not test multiple conformations)
    "n_symmetries": 10, # Total metal-center symmetrys to build, NSymmetries should be >= n_conformers
    "relax": True, # Perform final geomtetry relaxation of assembled complexes
    "save_init_geos": False, # Save initial geometries before relaxations.
    "crest_sampling": False, # Perform CREST sampling on lowest-energy conformer before returning.
    "return_timings": True, # Return all intermediate and final timings.
    "skip_duplicate_tests": False, # Skip the duplicate tests (return all generated/relaxed configurations)
    "return_full_complex_class": False, # Return the complex class containing all ligand geometry and core information.
    "uid": u_id, # Unique ID (generated by default, but can be assigned)
    "seed": None, # If a seed is passed (int/float) use it to initialize np.random.seed for reproducability.
    # If you want to replicate whole workflows - set np.random.seed() at the beginning of your workflow.
    # Right not openbabel will still introduce randomness into generations - so it is often valuable
    # To run multiple searches if something is failing.

    # Dump all possible intermediate xtb calculations to separate ASE database
    "dump_ase_atoms": False, # or True
    "ase_atoms_db_name": 'architector_ase_db_{uid}.json', # Possible to name the databse filename
    # Will default to a "uid" included name.
    "temp_prefix":"/tmp/", # Default here - for MPI running on HPC suggested /scratch/$USER/

    ####### Ligand parameters #########
    # Ligand to finish filling out coordination environment if underspecified.
    "fill_ligand": "water", 
    # Secondary fill ligand will be a monodentate ligand to fill out coordination environment
    # in case the fill_ligand and specified ligands list cannot fully map to the coordination environment.
    "secondary_fill_ligand": "water",
    # or integer index in reference to the ligand list!!
    "force_trans_oxos":False, # Force trans configurations for oxos (Useful for actinyls)
    "lig_assignment":'bruteforce', # or "similarity" - How to automatically assign ligand types.

    ######### Sanity check parameters ########
    "assemble_sanity_checks":True, # Turn on/off assembly sanity checks.
    "assemble_graph_sanity_cutoff":1.8, 
    # Graph Sanity cutoff for imposed molecular graph represents the maximum elongation of bonds
    # rcov1*full_graph_sanity_cutoff is the maximum value for the bond lengths.
    "assemble_smallest_dist_cutoff":0.3,
    # Smallest dist cutoff screens if any bonds are less than smallest_dist_cutoff*sum of cov radii
    # Will not be evaluated by XTB if they are lower.
    "assemble_min_dist_cutoff":4,
    # Smallest min dist cutoff screens if any atoms are at minimum min_dist_cutoff*sum of cov radii
    # away from ANY other atom (indicating blown-up structure) 
    # - will not be evaluated by XTB if they are lower.
    "full_sanity_checks":True, # Turn on/off final sanity checks.
    "full_graph_sanity_cutoff":1.7,
    # full_graph_sanity_cutoff can be tightened to weed out distorted geometries (e.g. 1.5 for non-group1-metals) 
    "full_smallest_dist_cutoff":0.55,
    "full_min_dist_cutoff":3.5,
    } 
}
```

## Output dictionary structure and recommendations:
```python
out = {
    'core_geometry_i_nunpairedes_X_charge_Y': 
    # Key labels indicates metal center geometry, total unpaired electrons (X, spin), 
    # and charge (Y) of the complex
    {'ase_atoms':ase.atoms.Atoms, # Structure (with attached used ASE calculator!) for the output complex.
    'total_charge': int, # Suggested total charge if another method used.
    'calc_n_unpaired_electrons': int, # Suggested unpaired electrons if another method used.
    'xtb_total_charge':int, # Same as (Y) (different from total_charge for non-oxidation state=3 f-block elements!)
    'xtb_n_unpaired_electrons' : int, # Same as (X) Unpaired electrons used for xTB (different for f-block elements!) 
    'metal_ox': int, # Metal oxidation state assigned to the complex
    'init_energy': float, # Initial (unrelaxed) xTB energy (eV)
    'energy': float, # Relaxed xTB energy (eV)
    'mol2string': str, # Final relaxed structure in TRIPOS mol2 format.
    'init_mol2string': str, # Initial unrelaxed structure in TRIPOS mol2 format.
    'energy_sorted_index': int, # Index of the complex from pseudo-energy ranking,
    'inputDict': dict, # Full input dictionary copy (including assigned parameters) for replication!
    ..... Timing information ....},
    ** More structures **
}
```

* Note that output dictionary is an OrderDict sorted by energy (first entry is the lowest in energy.)

Within the jupyter notebook framework it is quite easy to visualize all of the generated structures directly from the dictionary:
```python
from architector import view_structures

view_structures(out)
```

With the following example line it is quite easy to export to xyz for use in any other electronic structure code:
```python
out['core_geometry_i_nunpairedes_X_charge_Y']['ase_atoms'].write('core_geometry_i_nunpairedes_X_charge_Y.xyz')
```

Alternatively, a file format converter is included with Architector which can read the formatted mol2 filetypes,
which can be quite useful for maintaining the defined molecular graph and bond orders:

```python
from architector import convert_io_molecule

mol = convert_io_molecule(out['core_geometry_i_nunpairedes_X_charge_Y']['mol2string'])
print(mol.uhf) # n_unpaired electrons for electronic structure evaluation
print(mol.charge) # total charge
mol.write_xyz('core_geometry_i_nunpairedes_X_charge_Y.xyz')
```

Included in the Architector Molecule python object (mol, above) is also the molecular graph (mol.graph) and SYBYL type bond orders (mol.BO_dict),
xTB unpaired electrons (mol.xtb_uhf), and a full ASE Atoms object (mol.ase_atoms) with the assigned charge (mol.charge) and magnetic moments (mol.uhf)
from the output mol2string.

## Authors:

* Michael G. Taylor
* Daniel J. Burrill
* Jan Janssen 
* Danny Perez
* Enrique R. Batista
* Ping Yang

## Licensing and Copyright: 

See LICENSE.txt for licensing information. Architector is licensed under the BSD-3 license.
The Los Alamos National Laboratory C Number for Architector is C22085.

© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.