"""
Basic Docking routine for solvent molecules around a central metal.

Developed by Michael Taylor
"""
import copy
import numpy as np
import architector.io_obabel as io_obabel
import architector.io_molecule as io_molecule
import architector.io_ptable as io_ptable
from scipy.spatial.transform import Rotation as Rot
from architector.io_calc import CalcExecutor
from ase import units

defaults = {
    # "Secondary Solvation Shell" parameters
    'species_list': ['water']*3,  # Pass a list of species (preferred)
    "freeze_molecule_add_species": False,  # Whether to free the original moleucule during all secondary
    # shell relaxations, default False.
    'species_smiles': 'O',  # Can also specify multiple copies of single species with this and next line
    'n_species': 3,  # -->> this line!
    'n_species_rotations': 20,  # Rotations in 3D of ligands to try
    'n_species_conformers': 1,  # Number of conformers to try - right now only 1 will be tested.
    'species_grid_pad': 5,  # How much to pad around the molecule species are being added to (in Angstroms)
    'species_gridspec': 0.3,  # How large of steps in R3 the grid surrounding a molecule should be
    # to which a species could be added. (in Angstroms)
    'species_skin': 0.2,  # How much buffer or "skin" should be added to around a molecule 
    # to which the species could be added. (in Angstroms)
    'species_grid_rad_scale': 1,  # Factor to multiply molecule+species vdw by to set molecules.
    # e.g. Reduce to allow for closer molecule-species distances
    'species_location_method': 'default',  # Default attempts a basic colomb repulsion placement, targeted
    # Only other option is 'random' at the moment.
    'species_add_copies': 1,  # Number of species addition orientations to build 
    'species_method': 'GFN2-xTB',  # Method to use on full species - right now only GFN2-xTB really works
    'species_relax': True,  # Whether or not to relax the generated secondary solvation structures.
    'species_intermediate_method': 'GFN-FF',  # Method to use for intermediate species screening - Suggested GFN-FF
    'species_intermediate_relax': True,  # Whether to perform the relaxation only after all secondary species are added
    "calculator": None,  # ASE calculator class input for usage during construction or for optimization.
    "calculator_kwargs": dict(),  # ASE calculator kwargs.
    "ase_opt_method": None,  # ASE optimizer class used for geometry optimizations. Default will use LBFGSLineSearch.
    "ase_opt_kwargs": dict(),  # ASE optimizer kwargs.
    'targeted_indices_close': None,  # Indices of both base molecule and added molecule to add close to one another.
    # e.g. [1,0]
    'debug': False  # Debug
}

targeted_defaults = {
    # "Secondary Solvation Shell" parameters
    'species_list': ['water']*3,  # Pass a list of species (preferred)
    "freeze_molecule_add_species": False,  # Whether to free the original moleucule during all secondary
    # shell relaxations, default False.
    'species_smiles': 'O',  # Can also specify multiple copies of single species with this and next line
    'n_species': 3,  # -->> this line!
    'n_species_rotations': 200,  # Rotations in 3D of ligands to try
    'n_species_conformers': 1,  # Number of conformers to try - right now only 1 will be tested.
    'species_grid_pad': 5,  # How much to pad around the molecule species are being added to (in Angstroms)
    'species_gridspec': 0.1,  # How large of steps in R3 the grid surrounding a molecule should be
    # to which a species could be added. (in Angstroms)
    'species_skin': 0.6,  # How much buffer or "skin" should be added to around a molecule 
    # to which the species could be added. (in Angstroms)
    'species_grid_rad_scale': 1.2,  # Factor to multiply molecule+species vdw by to set molecules.
    # e.g. Reduce to allow for closer molecule-species distances
    'species_location_method': 'targeted',  # Default attempts a basic colomb repulsion placement, targeted
    # Only other option is 'random' at the moment.
    'species_add_copies': 5,  # Number of species orientations to build 
    'species_method': 'GFN2-xTB',  # Method to use on full species - right now only GFN2-xTB really works
    'species_relax': False,  # Whether or not to relax the generated secondary solvation structures.
    'species_intermediate_method': 'GFN-FF',  # Method to use for intermediate species screening - Suggested GFN-FF
    'species_intermediate_relax': False,  # Whether to perform the relaxation only after all secondary species are added
    "calculator": None,  # ASE calculator class input for usage during construction or for optimization.
    "calculator_kwargs": dict(),  # ASE calculator kwargs.
    "ase_opt_method": None,  # ASE optimizer class used for geometry optimizations. Default will use LBFGSLineSearch.
    "ase_opt_kwargs": dict(),  # ASE optimizer kwargs.
    'targeted_indices_close': None,  # Indices of both base molecule and added molecule to add close to one another.
    # e.g. [1,0]
    'debug': False  # Debug
}


def center_molecule_gen_grid(mol, parameters={}):
    """center_molecule_gen_grid
    Put the molecule in the center of a box by with N angstroms of padding on each max/min xyz.
    Use the "species_gridspec" and "species_grid_pad" to determine the shape of the mesh.

    Parameters
    ----------
    mol : architector.io_molcule.Molecule
        Architector Molecule
    parameters : dict, optional
        Parameters for species, by default {}

    Returns
    -------
    outgrid, np.ndarray
        Nx3 Grid of points in [x,y,z] corresponding to mesh points.
    """
    coords = mol.ase_atoms.get_positions()  # Get positions
    new_coords = coords - coords.mean(axis=0)  # Move to 0,0,0 as center of geometry
    mol.ase_atoms.set_positions(new_coords)  # In the middle of the box.
    mins = new_coords.min(axis=0) - parameters['species_grid_pad']  # Add padding
    maxs = new_coords.max(axis=0) + parameters['species_grid_pad']  # Add padding
    x_ = np.arange(mins[0], maxs[0], parameters['species_gridspec'])
    y_ = np.arange(mins[0], maxs[0], parameters['species_gridspec'])
    z_ = np.arange(mins[0], maxs[0], parameters['species_gridspec'])
    grid = np.meshgrid(x_, y_, z_, indexing='ij')
    mgrid = list(map(np.ravel, grid))
    outgrid = np.vstack(mgrid).T  # Flatten meshgrid to nx3
    return outgrid


def get_rad_effective(mol):
    """get_rad_effective
    Calculate the effective radius of the molecule given.

    Parameters
    ----------
    mol : architector.molecule.Molecule
        Molecule to calclulate the effective radius

    Returns
    -------
    out : tuple
        (mean distance from centroid, max distance from centroid)
    """
    coords = mol.ase_atoms.get_positions()
    new_coords = coords - coords.mean(axis=0)
    mol.ase_atoms.set_positions(new_coords)
    dists = np.linalg.norm(new_coords, axis=1)
    out = (np.mean(dists), np.max(dists))
    return out


def species_generate_get_ref_params(species_id,
                                    parameters={},
                                    main_molecule=False,
                                    intermediate=False):
    """species_generate_get_ref_params
    Get charges and species information for the generated species.

    Parameters
    ----------
    species_id : str (other)
        The identity of the species, either a smiles str, mol2string, or xyzstring,
        or architector.io_molecule.Molecule
    parameters : dict, optional
        parameters to generate, by default {}
    main_molecule : bool, optional
        Whether this is the central molecule to add other molecules to, by default False
    intermediate : bool, optional
        Whether this is an intermediate calculation or not, by default False


    Returns
    -------
    species, overloaded architector.io_molecule.Molecule
        Architector molecule object with species with attached .param_dict[] object.
    """
    species = io_molecule.convert_io_molecule(species_id)
    # Possibly generate a bunch more conformers from a SMILES?
    # species_confs = io_obabel.generate_obmol_conformers(solvent_smi)
    if isinstance(species_id, str):
        # Not a structure passed.
        if ('TRIPOS' in species_id) or (
           species_id.splitlines()[0].strip().isnumeric()):
            outdict = {'charge': species.charge, 'uhf': species.uhf}
        else:
            try:
                smi_obmol = io_obabel.get_obmol_smiles(species_id)
                outdict = {'charge': smi_obmol.GetTotalCharge()}
                outdict.update({'uhf': smi_obmol.GetTotalSpinMultiplicity() - 1})
            except:
                outdict = {'charge': species.charge,
                           'uhf': species.uhf}
    else:  # Otherwise these should be assigned to the molecule
        outdict = {'charge': species.charge, 
                   'uhf': species.uhf}
    mean_rad, max_rad = get_rad_effective(species)
    if any([(species.charge is None),
            (species.uhf is None),
            (species.xtb_charge is None),
            (species.xtb_uhf is None)]):
        species.calc_suggested_spin()  # Use molecule spin/charge detection.
    outdict.update({'mean_rad': mean_rad,
                    'max_rad': max_rad})
    if parameters.get('species_location_method', 'default') != 'random':
        calc = CalcExecutor(species,
                            parameters=parameters,
                            species_run=True,
                            intermediate=intermediate)
        species = calc.mol
        outdict['species_dipole'] = species.ase_atoms.calc.results.get('dipole',
                                                                       np.zeros(3))
        outdict['species_dipole_mag'] = np.linalg.norm(outdict.get('species_dipole',
                                                                   None))
        outdict['species_charges'] = species.ase_atoms.calc.results.get('charges',
                                                                        np.zeros(len(calc.mol.ase_atoms)))
        outdict['energy'] = species.ase_atoms.calc.results.get('energy',
                                                               100000)
        outdict['forces'] = species.ase_atoms.calc.results.get('forces',
                                                               np.zeros((len(calc.mol.ase_atoms),3)))
    else:
        calc = CalcExecutor(species,
                            parameters=parameters,
                            species_run=True,
                            intermediate=intermediate)
        species = calc.mol
        outdict['energy'] = calc.energy
    if not main_molecule:
        rotations_lst = []
        species.ase_atoms.calc = None
        for _ in range(parameters['n_species_rotations']):
            q = Rot.random()
            temp = copy.deepcopy(species.ase_atoms)
            temp.set_positions(q.apply(temp.positions))
            rotations_lst.append(temp)
        outdict['rotations_list'] = rotations_lst
    # Save param_dict to species.
    setattr(species, 'param_dict', outdict)
    return species


def decide_new_species_location(mol, species, parameters={}):
    """decide_new_species_location 
    Select where the species should be added.
    Method can be changed with parameters['species_location_method'].
    Current options are only:
    1. "default" which does a basic colomb approximation.
    2. "random" which selects a random "valid" position.

    Parameters
    ----------
    mol : architector.molecule.Molecule
        Central molecule on which to add the species
        (overloaded with .param_dict object)
    species : architector.molecule.Molecule
        Species molecule to add to molecule
        (overloaded with .param_dict object)
    parameters : dict, optional
        Generation parametrs, by default {}

    Returns
    -------
    out_location : np.ndarray
        position selected for where it should sit
    """
    grid = center_molecule_gen_grid(mol, parameters=parameters)
    spec_rad = species.param_dict['mean_rad']
    dist_grid_molecule = np.linalg.norm(mol.ase_atoms.get_positions()[:, None, :] - \
                           grid[None, :, :],axis=-1)
    upper_radvect = []
    lower_radvect = []
    for z in mol.ase_atoms.get_atomic_numbers():
        grid_radii = parameters.get('species_grid_rad_scale',
                                    1)*(io_ptable.rvdw[z]+spec_rad)
        upper_radvect.append(grid_radii+parameters['species_skin'])
        lower_radvect.append(grid_radii)
    upper_radvect = np.array(upper_radvect).reshape(-1, 1)
    lower_radvect = np.array(lower_radvect).reshape(-1, 1)
    upper_inds = np.where(np.any(np.less_equal(dist_grid_molecule,
                                               upper_radvect), axis=0))[0]
    lower_inds = np.where(np.any(np.less_equal(dist_grid_molecule,
                                               lower_radvect), axis=0))[0]
    shared_inds = np.setdiff1d(upper_inds, lower_inds)
    if parameters['species_location_method'] == 'random':
        sel_ind = np.random.choice(shared_inds, size=1)
        out_location = grid[sel_ind]
    elif parameters['species_location_method'] == 'default':
        if parameters['debug']:
            print(
                'Shape Check on Viable inds, All gridpoints , and All griddists.')
            print(shared_inds.shape,
                  grid.shape,
                  dist_grid_molecule.T.shape)
        possible_grid_locs = grid[shared_inds]
        possible_grid_dists = dist_grid_molecule.T[shared_inds]
        spec_charge = species.param_dict['charge']
        pick_mag = False
        if spec_charge > 0:  # Add in species molecular dipole to "charge"
            spec_charge = spec_charge + species.param_dict['species_dipole_mag']
        elif spec_charge < 0:
            spec_charge = spec_charge - species.param_dict['species_dipole_mag']
        else:
            spec_charge = species.param_dict['species_dipole_mag']
            pick_mag = True
        if parameters['debug']:
            print('Spec Charge:', spec_charge)
        eq = 1/(4*np.pi*units._eps0) * spec_charge * mol.param_dict['species_charges'] \
            / (possible_grid_dists)  # kqQ/r
        eq = np.sum(eq, axis=1)
        if parameters['debug']:
            print('eq Energy: Max: ',
                  np.max(eq),'min:',
                  np.min(eq),'std:',
                  np.std(eq))
        if pick_mag:
            out_location = possible_grid_locs[np.argmax(np.abs(eq))]
        else:
            out_location = possible_grid_locs[np.argmin(eq)]
    elif (parameters['species_location_method'] == 'targeted') and \
         (isinstance(parameters['targeted_indices_close'],
                     (list, tuple, np.ndarray))):
        target_inds = parameters['targeted_indices_close']
        upper_inds_target = np.where(np.less_equal(dist_grid_molecule[target_inds[0]], 
                                                   upper_radvect[target_inds[0]]))[0] # Get grid points close to only this atom
        lower_inds_target = np.where(np.less_equal(dist_grid_molecule[target_inds[0]], 
                                                   lower_radvect[target_inds[0]]))[0] # Get grid points close to only this atom
        shared_inds_target = np.setdiff1d(upper_inds_target,lower_inds_target)
        shared_inds_sel = np.intersect1d(shared_inds_target,shared_inds)
        if parameters['debug']:
            print('Total possible locations',len(shared_inds_sel),len(shared_inds_target))
        sel_ind = np.random.choice(shared_inds_sel,size=1)
        out_location = grid[sel_ind]
    return out_location


def add_species(init_mol, species, parameters={}):
    """add_species 
    Add a species to the central "init_mol"

    Parameters
    ----------
    init_mol : architector.io_molecule.Molecule
        Central molecule to which to add another molecule
        (overloaded with .param_dict object)
    species : architector.io_molecule.Molecule
        Species to add to central molecule 
        (overloaded with .param_dict object)
    parameters : dict, optional
        Parameters to use, by default {}

    Returns
    -------
    newmol : architector.io_molecule.Molecule
        Molecule with species added.
    """
    spec_loc = decide_new_species_location(init_mol,
                                           species,
                                           parameters=parameters)
    tmp_spec = copy.deepcopy(species)
    init_mol.ase_atoms.calc = None
    rotations = tmp_spec.param_dict['rotations_list']
    best_obj = np.inf
    out_rotation = None
    for i, r in enumerate(rotations):  # Test all rotations using 'GFN-FF'
        if parameters['debug']:
            print('Trying rotation {}'.format(i))
        tmp_mol = copy.deepcopy(init_mol)
        tmp_spec.ase_atoms.set_positions(spec_loc - r.get_positions())
        spec_dict = {'bo_dict': tmp_spec.BO_dict,
                     'ase_atoms': tmp_spec.ase_atoms,
                     'atom_types': tmp_spec.atom_types,
                     'uhf': tmp_spec.uhf,
                     'xtb_uhf': tmp_spec.xtb_uhf,
                     'charge': tmp_spec.charge,
                     'xtb_charge': tmp_spec.xtb_charge}
        tmp_mol.append_ligand(spec_dict,
                              non_coordinating=True)
        tmp_mol.dist_sanity_checks()
        if tmp_mol.dists_sane:
            calc = CalcExecutor(tmp_mol,
                                parameters=parameters,
                                species_run=True,
                                intermediate='rotation')
            if parameters['species_location_method'] in ['default','random']:
                obj = calc.energy
            elif parameters['species_location_method'] in ['targeted']:
                target_inds = parameters['targeted_indices_close']
                targeted_dist = calc.mol.ase_atoms.get_distance(target_inds[0],
                                                                len(init_mol.ase_atoms)+target_inds[1])
                obj = calc.energy / targeted_dist  # Both lower energy (-VE), lower distance (+VE)
            if obj < best_obj:
                out_rotation = calc.mol
                best_obj = obj
    good = False
    if out_rotation is not None:
        good = True
        newmol = species_generate_get_ref_params(out_rotation,
                                                 parameters=parameters,
                                                 main_molecule=True,
                                                 intermediate='main')
    else:
        if parameters['debug']:
            print('None of the Ligands passed the calculator addition - skipping ONE.')
        good = False
        newmol = tmp_mol
    return newmol, good


def add_non_covbound_species(mol, parameters={}):
    """add_non_covbound_species 
    Use basic docking techniques to add a given number of defined species to a central
    Molecule:
    parameters = {
        # "Secondary Solvation Shell" parameters
        "species_list": ['water']*3, # Pass a list of species (preferred)
        "freeze_molecule_add_species": False, # Whether to free the original moleucule during all secondary
        # shell relaxations, default False.
        "species_smiles": "O", # Can also specify multiple copies of single species with this and next line
        "n_species": 3, # -->> this line!
        "n_species_rotations": 20, # Rotations in 3D of ligands to try
        "n_species_conformers": 1, # Number of conformers to try - right now only 1 will be tested.
        "species_grid_pad": 5, # How much to pad around the molecule species are being added to (in Angstroms)
        "species_gridspec": 0.3, # How large of steps in R3 the grid surrounding a molecule should be
        # to which a species could be added. (in Angstroms)
        "species_skin": 0.2, # How much buffer or "skin" should be added to around a molecule 
        # to which the species could be added. (in Angstroms)
        "species_grid_rad_scale": 1, # Factor to multiply molecule+species vdw by to set molecules.
        # e.g. Reduce to allow for closer molecule-species distances
        "species_location_method": "default", # 'default' attempts a basic colomb repulsion placement.
        # Another option is 'random' at the moment. Finally 'targeted' attempts to pick locations and rotations
        # Optimizing for two indices (one from mol and 1 from the species added) to be close. 
        # See "targeted_indices_close".
        "species_add_copies": 1, # Number of species addition orientations to build 
        "species_method": "GFN2-xTB", # Method to use on full species - right now only GFN2-xTB really works
        "species_relax": True, # Whether or not to relax the generated secondary solvation structures.
        "species_intermediate_method": "GFN-FF", # Method to use for intermediate species screening - Suggested GFN-FF
        "species_intermediate_relax": True, # Whether to perform the relaxation only after all secondary species are added
        "calculator": None, # ASE calculator class input for usage during construction or for optimization.
        "calculator_kwargs": dict(), # ASE calculator kwargs.
        "ase_opt_method": None, # ASE optimizer class used for geometry optimizations. Default will use LBFGSLineSearch.
        "ase_opt_kwargs": dict(), # ASE optimizer kwargs.
        "ase_opt_kwargs": {}, # ASE optimizer
        "targeted_indices_close": None, # Indices of both base molecule and added species to add close to one another.
        # e.g. [1,0]
        "debug": False # Debug
    }
    
    Parameters
    ----------
    mol : architector.io_molecule.Molecule 
        Structure to add additional ligands to!
    parameters : dict, optional
        See above, by default {}

    Returns
    -------
    init_mol : architector.io_molecule.Molecule
        structure with species added to the molecule

    Raises
    ------
    ValueError
        Needs either "species_list" or "n_species"/"species_smiles" specified.
    """
    if parameters.get('species_location_method','default') in ['default','random']:
        params = copy.deepcopy(defaults)
    elif parameters.get('species_location_method','default') in ['targeted']:
        params = copy.deepcopy(targeted_defaults)
    params.update(parameters)
    species_list = params.get('species_list',None)
    n_species = params.get('n_species',None)
    species_smiles = params.get('species_smiles',None)
    if (species_list is not None):
        species_list = [io_ptable.ligands_dict.get(x,{'smiles':x})['smiles'] for x in species_list]
    elif (species_smiles is not None) and (n_species is not None):
        species_list = [io_ptable.ligands_dict.get(x,{'smiles':x})['smiles'] for x in [species_smiles]*n_species]
    else:
        raise ValueError('Need either "species_list" specified OR "n_species" and "species_smiles" specified.')
    unique_specs = list(set(species_list))
    species_add_list = []
    n = params.get("species_add_copies", 1)
    for j in range(n):
        species_dict = dict()
        for spec in unique_specs:
            species = species_generate_get_ref_params(spec,
                                                      parameters=params)
            species_dict[spec] = species
        if params.get('debug',False):
            print('Doing all species random addition {} of {}'.format(j+1,n))
        init_mol = species_generate_get_ref_params(mol,
                                                   parameters=params,
                                                   main_molecule=True)
        good = True
        for i, spec in enumerate(species_list):
            if params['debug']:
                print('Adding species {} of {}.'.format(i+1,len(species_list)))
                print(species_dict[spec].write_mol2('cool_species',
                                                    writestring=True))
            init_mol, good = add_species(
                init_mol,
                species_dict[spec],
                parameters=params)
        # Ensure the last configuration is relaxed if requested.
        if params.get('species_relax', True) and \
           (not params.get('species_intermediate_relax', False)) and good:
            out_mol = species_generate_get_ref_params(
                init_mol,
                parameters=params,
                main_molecule=True)
            species_add_list.append(out_mol)
        elif not good:
            pass
        else:  # Just return final molecule.
            out_mol = init_mol
            species_add_list.append(out_mol)
    if len(species_add_list) > 0:
        outmol = species_add_list[
            np.argmin([x.param_dict["energy"] for x in species_add_list])]
        return outmol, species_add_list
    else:
        return mol, []

    