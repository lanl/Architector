""" 
Code for aligning molecules with arbitrary order. (Allowing permutations)
Applies by default to core of graoh depth 3 from the metal to align molecules.

Relies on a couple algorithms:
https://en.wikipedia.org/wiki/Hungarian_algorithm 
https://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions
https://en.wikipedia.org/wiki/Kabsch_algorithm 
A lot of shared ideas from:
https://pypi.org/project/rmsd/ 


Developed by Michael Taylor
"""
from multiprocessing.sharedctypes import Value
import numpy as np
import scipy
import copy

from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot
from architector import io_ptable

from architector.io_molecule import (convert_io_molecule, convert_ase_xyz)

def get_graph_depths(graph):
    """get_graph_depths 
    calculate all graph depths from molecular graph

    Returns
    -------
    depths : np.ndarray
        NXN matrix containing the graph depths of each atom to the others.
    """
    depths = scipy.sparse.csgraph.dijkstra(scipy.sparse.csgraph.csgraph_from_dense(graph))
    return depths

#### Permutation ordering and rmsd loss assignment adapted LANL code developed by Chang Liu

def permutation_cost_mat(pt_list1, pt_list2, label1, label2, costtype='xyz'):
    """permutation_cost_mat 
    return the cost matrix required by Hungarian method
    
    Only permutation with same label is allowed

    Parameters
    -------
    pt_list1 : np.ndarray
        array of coordinates of 1 
    pt_list2 : np.ndarray
        array of coordinates of 2 
    label1 : list/np.ndarray
        atom types or symols tied to coordinates 1
    label2 : list/np.ndarray
        atom types or symols tied to coordinates 2
    costtype : str, optional
        what type of cost/distance loss to use, by default 'xyz'
        'xyz' - pairwise distance from point of 1 to point in 2
        'COM' - distance to the center of mass

    Returns
    -------
    cost_mat : np.ndarray
        2D array NXN containing costs for the hungarian method
    """
    npt = len(pt_list1)
    cost_mat = np.zeros([npt, npt])
    pt_list1_com = np.mean(pt_list1,axis=0)
    pt_list2_com = np.mean(pt_list2,axis=0)
    for i, (xyz1, label1_t) in enumerate(zip(pt_list1, label1)):
        for j, (xyz2, label2_t) in enumerate(zip(pt_list2, label2)):
            if label1_t != label2_t:
                # permutation between different element is not allowed
                cost_mat[i,j] = np.inf
            else:
                if costtype == 'xyz':
                    diffvec = xyz1 - xyz2
                elif costtype == 'COM':
                    diffvec = (xyz1-pt_list1_com) - (xyz2-pt_list2_com)
                cost_mat[i,j] = np.dot(diffvec, diffvec)
    return cost_mat


def rmsd_align(tarmol, insrcmol, in_place=False):
    """rmsd_align 
    Perform alignment between two ase Atoms assuming indices are ideally permuted
    Return the rmsd_loss for the rotation.

    Parameters
    ----------
    tarmol : ase.atoms.Atoms
        reference/target molecule to match the generated molecule to.
    insrcmol : ase.atoms.Atoms
        generated/predicted molecule to match to reference/target molecule.
    in_place : bool, optional
        [description], by default False

    Returns
    -------
    rmsd_loss : float
        rmsd loss of the rotated version.
    r : scipy.spatial.transform.Rotation
        rotation instance to replicate matching the vectors.
    insrcmol : ase.atoms.Atoms
        rotated instance of the input molecule
    """
    insrcmol = insrcmol.copy()
    if not in_place:
        r = Rot.align_vectors(tarmol.positions, insrcmol.positions)
        newposits = r[0].apply(insrcmol.positions)
        insrcmol.set_positions(newposits)
    else: 
        r = None
    rmsd_loss = np.sqrt(np.sum((insrcmol.positions-tarmol.positions)**2)/len(tarmol))
    return rmsd_loss, r, insrcmol


def permute_align(tarmol, srcmol, maxiter=1, tol=1e-6, in_place=False):
    """permute_align 
    permute the atom order in mol to minimize the rmsd, in place

    A follow up rmsd alignment will be applied to the new molecule to
    calculate the new rmsd

    Parameters
    ----------
    tarmol : ase.atoms.Atoms
        reference/target molecule to match the generated molecule to.
    srcmol : ase.atoms.Atoms
        generated/predicted molecule to match to reference/target molecule.
    maxiter : int, optional
        maximum times permutation/rotation is repeated, by default 1
    tol : float, optional
        convergence tolerance, by default 1e-6
    in_place : bool, optional
        whether to perform permute with no rotation to calculate rsmd, by default False

    Returns
    -------
    rms : float
        rmsd of the permutation/rotation
    outr : scipy.spatial.transform.Rotation
        rotation instance required to align the molecules
    srcmol_1 : ase.atoms.Atoms
        rotated source molecule.
    """
    tarmol_1 = tarmol.copy()
    srcmol_1 = srcmol.copy()
    last_rms = np.inf
    outr = None
    count = 0
    for _ in range(maxiter):
        cost_mat = permutation_cost_mat(tarmol_1.positions, srcmol_1.positions, 
                                        tarmol_1.get_atomic_numbers(), srcmol_1.get_atomic_numbers(),
                                       costtype='xyz')
        permute = linear_sum_assignment(cost_mat)[1]
        srcmol_1 = srcmol_1[permute]
        rms, r, srcmol_1 = rmsd_align(tarmol_1, srcmol_1, in_place=in_place)
        if (count == 0) and (not in_place):
            outr = r[0] 
        elif (not in_place):
            outr = outr * r[0] # Composite rotation.
        else:
            outr = r
        if abs(rms-last_rms) < tol:
            break
        last_rms = rms
        count += 1
    return rms, outr, srcmol_1


def mirror_align(tarmol, srcmol, maxiter=1, tol=1e-6):
    """mirror_align 
    try mirror image alignment of molecule.

    Parameters
    ----------
    tarmol : ase.atoms.Atoms
        target molecule
    srcmol : ase.atoms.Atoms
        source molecule
    maxiter : int, optional
        how many times to try, by default 10
    tol : float, optional
        tolerance for convergence, by default 1e-2

    Returns
    -------
    rmsd : float
        rmsd value of converged mirror permutation
    outr : scipy.spatial.transform.Rotation
        rotation instance of best mirror image rotation.
    msrcmol : ase.atoms.Atoms
        mirrored rotated version of the molecule.
    """
    srcmol_tmp = srcmol.copy()
    newposits = srcmol_tmp.positions
    newposits[:,0] = -newposits[:,0]
    srcmol_tmp.set_positions(newposits)
    rmsd, outr, msrcmol = permute_align(tarmol,srcmol_tmp,maxiter=maxiter,tol=tol)
    return rmsd, outr, msrcmol
    

def calc_rmsd(genMol, compareMol, coresize=2, maxiter=1, sample=300, atom_types=None,
              return_structures=False, rmsd_type='simple'): 
    """calc_rmsd 
    Calculate the rmsd by different methods for this molecule compared to another.

    Parameters
    ----------
    genMol : str/Molecule
        molecule generated to rotate to reference.
    compareMol : str/Molecule
        molecule to compare to (reference molecule)
    rmsd_type : str, optional
        which type of rmsd to calculate , by default 'simple', possible 'edge'
    coresize : int, optional
        number of graph hops from the metal to consider when aligning molecules, by default 3 
    maxiter : int, optional
        number of iterations per time on permutation/alignment routine, by default 1
    sample : int, optional
        number of random rotations to sample initially to ensure best mapping between cores, by default 30
    atom_types: list, optional
        which atom types to consider for alignment - will default to full alignment, default None
    return_structures : bool, optional
        whether or not to return the rotated versions of the core and full structures, by defualt False

    Returns
    -------
    rmsd : float
        RMSD per atom comparison between the molecules.
    """

    genMol = convert_io_molecule(genMol)
    compareMol = convert_io_molecule(compareMol)

    # Check that these are stoichiometrically identical molecules.
    if np.any(sorted(genMol.ase_atoms.get_atomic_numbers()) != sorted(compareMol.ase_atoms.get_atomic_numbers())):
        print('Warning - comparison not possible between molecules of different sizes/stoichiometries.')
        flag_struct = True
        # Set ordering to be identical based on canonical labels.
        genMol_metalind = genMol.find_metal()
        genMol_graph_depths = get_graph_depths(genMol.graph)

        compareMol_metalind = compareMol.find_metal()
        compareMol_graph_depths = get_graph_depths(compareMol.graph)

        # Pull out center of molecule up to depth coresize graph hops for matching to reference.
        genMol_subset_component_inds = np.where(genMol_graph_depths[genMol_metalind] <= coresize)[0]
        compareMol_subset_component_inds = np.where(compareMol_graph_depths[compareMol_metalind] <= coresize)[0]

        tmp_self_comp = genMol.ase_atoms[genMol_subset_component_inds].copy() 
        tmp_ref_comp = compareMol.ase_atoms[compareMol_subset_component_inds].copy() 
        outcore = tmp_self_comp
        rmsd_loss_core = 1000
        rmsd_loss_full = 1000
    else:
        # Set ordering to be identical based on canonical labels.
        genMol_metalind = genMol.find_metal()
        genMol_graph_depths = get_graph_depths(genMol.graph)

        compareMol_metalind = compareMol.find_metal()
        compareMol_graph_depths = get_graph_depths(compareMol.graph)

        # Pull out center of molecule up to depth coresize graph hops for matching to reference.
        genMol_subset_component_inds = np.where(genMol_graph_depths[genMol_metalind] <= coresize)[0]
        compareMol_subset_component_inds = np.where(compareMol_graph_depths[compareMol_metalind] <= coresize)[0]

        tmp_self_comp = genMol.ase_atoms[genMol_subset_component_inds].copy() 
        tmp_ref_comp = compareMol.ase_atoms[compareMol_subset_component_inds].copy() 

        flag_struct = False

        if np.any(sorted(tmp_self_comp.get_atomic_numbers()) != sorted(tmp_ref_comp.get_atomic_numbers())):
            print('Warning cores not matched at graph depth!')
            print('Can re-attempt. Not doing that now.')
            print(tmp_self_comp.get_chemical_formula())
            print(tmp_ref_comp.get_chemical_formula())
            outcore = tmp_self_comp
            rmsd_loss_core = 1000
            rmsd_loss_full = 1000
            flag_struct = True
        else:
            # Center on metal atom
            tmp_self_comp.set_positions(tmp_self_comp.positions - genMol.ase_atoms[genMol_metalind].position)
            tmp_ref_comp.set_positions(tmp_ref_comp.positions - compareMol.ase_atoms[compareMol_metalind].position)
            genMol.ase_atoms.set_positions(genMol.ase_atoms.positions - genMol.ase_atoms[genMol_metalind].position)
            compareMol.ase_atoms.set_positions(compareMol.ase_atoms.positions - compareMol.ase_atoms[compareMol_metalind].position)
            
            # Sample random rotations to find best starting assignment point.
            best = np.inf
            for _ in range(sample):
                q = Rot.random()
                calc_test_comp = copy.deepcopy(tmp_self_comp)
                calc_test_comp.set_positions(q.apply(calc_test_comp.positions))
                rmsd_core, _, _ = permute_align(tmp_ref_comp, calc_test_comp, maxiter=maxiter)
                rmsd_mirror, _, _ = mirror_align(tmp_ref_comp, calc_test_comp, maxiter=maxiter)
                if rmsd_mirror < rmsd_core:
                    rmsd_core = rmsd_mirror
                if rmsd_core < best:
                    saveq = q
                    best = rmsd_core
                
            tmp_self_comp.set_positions(saveq.apply(tmp_self_comp.positions))
            genMol.ase_atoms.set_positions(saveq.apply(genMol.ase_atoms.positions))
            
            rmsd_core, r, outcore = permute_align(tmp_ref_comp, tmp_self_comp, maxiter=maxiter)
            rmsd_mirror, r_mirror, moutcore = mirror_align(tmp_ref_comp, tmp_self_comp, maxiter=maxiter)

            if rmsd_mirror < rmsd_core: # Pick the better one!
                rmsd_core = rmsd_mirror
                outcore = moutcore
                r = r_mirror
                newposits = genMol.ase_atoms.positions
                newposits[:,0] = -newposits[:,0] # Mirror across x axis to replicate mirror in permute
                genMol.ase_atoms.set_positions(newposits)
                    
            rmsd_loss_core = rmsd_core
            tmp_posits = genMol.ase_atoms.positions
            tmp_posits = r.apply(tmp_posits)
            genMol.ase_atoms.set_positions(tmp_posits)
        ## Do permutation mapping to estimate full loss given the rotation to match the core.
            rmsd_loss_full, _, _ = permute_align(copy.deepcopy(compareMol.ase_atoms), copy.deepcopy(genMol.ase_atoms), 
                                                    maxiter=1, tol=1e-6, in_place=True)
        
    if rmsd_type == 'simple':
        # Returns per-atom RMSD for ideal/rotation/translation overlap.
        if return_structures:
            return (rmsd_loss_core, 
                    rmsd_loss_full, 
                    compareMol.write_mol2('refmol.mol2',writestring=True), 
                    genMol.write_mol2('aligned.mol2',writestring=True),
                    convert_ase_xyz(tmp_ref_comp),
                    convert_ase_xyz(outcore),
                    flag_struct)
        else:
            return (rmsd_loss_core, rmsd_loss_full, flag_struct)
    else:
        print('Not yet implemented.')
        return None


def calc_rmsd_atypes(genMol, compareMol, sample=300, atom_types=None,
            rmsd_type='simple'): 
    """calc_rmsd 
    Calculate the rmsd by different methods for this molecule compared to another.

    Parameters
    ----------
    genMol : str/Molecule
        molecule generated to rotate to reference.
    compareMol : str/Molecule
        molecule to compare to (reference molecule)
    rmsd_type : str, optional
        which type of rmsd to calculate , by default 'simple', possible 'edge'
    sample : int, optional
        number of random rotations to sample initially to ensure best mapping between cores, by default 30
    atom_types: list, optional
        which atom types to consider for alignment - will default to full alignment, default None

    Returns
    -------
    rmsd : float
        RMSD per atom comparison between the molecules.
    """

    genMol = convert_io_molecule(genMol)
    compareMol = convert_io_molecule(compareMol)

    # Check that these are stoichiometrically identical molecules.
    # Set ordering to be identical based on canonical labels.
    if atom_types is None:
        compareMol_subset_component_inds = np.array([i for i,x in enumerate(compareMol.ase_atoms.get_chemical_symbols())])
        genMol_subset_component_inds = np.array([i for i,x in enumerate(genMol.ase_atoms.get_chemical_symbols())])
    elif isinstance(atom_types,str): 
        if atom_types == 'metals':
            compareMol_subset_component_inds = np.array([i for i,x in enumerate(compareMol.ase_atoms.get_chemical_symbols()) if x in io_ptable.all_metals])
            genMol_subset_component_inds = np.array([i for i,x in enumerate(genMol.ase_atoms.get_chemical_symbols()) if x in io_ptable.all_metals])
        elif atom_types == 'heavy_atoms':
            compareMol_subset_component_inds = np.array([i for i,x in enumerate(compareMol.ase_atoms.get_chemical_symbols()) if x != "H"])
            genMol_subset_component_inds = np.array([i for i,x in enumerate(genMol.ase_atoms.get_chemical_symbols()) if x != "H"])
        elif atom_types == 'heavy_metals':
            compareMol_subset_component_inds = np.array([i for i,x in enumerate(compareMol.ase_atoms.get_chemical_symbols()) if x in io_ptable.heavy_metals])
            genMol_subset_component_inds = np.array([i for i,x in enumerate(genMol.ase_atoms.get_chemical_symbols()) if x in io_ptable.heavy_metals])
        else:
            raise ValueError('I do not recognize this keyword "{}", please choose from "metals" or "heavy_atoms" or "heavy_metals"'.format(atom_types))
    else:
        compareMol_subset_component_inds = np.array([i for i,x in enumerate(compareMol.ase_atoms.get_chemical_symbols()) if x in atom_types])
        genMol_subset_component_inds = np.array([i for i,x in enumerate(genMol.ase_atoms.get_chemical_symbols()) if x in atom_types])

    # Pull out selected atoms from molecule up to depth coresize graph hops for matching to reference.
    tmp_self_comp = genMol.ase_atoms[genMol_subset_component_inds].copy() 
    tmp_ref_comp = compareMol.ase_atoms[compareMol_subset_component_inds].copy() 

    flag_struct = False

    if np.any(sorted(tmp_self_comp.get_atomic_numbers()) != sorted(tmp_ref_comp.get_atomic_numbers())):
        print('Warning subset stoichiometry not matched!')
        print('Can re-attempt. Not doing that now.')
        print(tmp_self_comp.get_chemical_formula())
        print(tmp_ref_comp.get_chemical_formula())
        outcore = tmp_self_comp
        rmsd_loss_core = 1000
        rmsd_loss_full = 1000
        flag_struct = True
    else:
        # Center on metal atom
        tmp_self_comp.set_positions(tmp_self_comp.positions - tmp_self_comp.get_positions().mean(axis=0))
        tmp_ref_comp.set_positions(tmp_ref_comp.positions - tmp_ref_comp.get_positions().mean(axis=0))
        genMol.ase_atoms.set_positions(genMol.ase_atoms.positions - tmp_self_comp.get_positions().mean(axis=0))
        compareMol.ase_atoms.set_positions(compareMol.ase_atoms.positions - tmp_ref_comp.get_positions().mean(axis=0))
        
        # Sample random rotations to find best starting assignment point.
        best = np.inf
        for _ in range(sample):
            q = Rot.random()
            calc_test_comp = copy.deepcopy(tmp_self_comp)
            calc_test_comp.set_positions(q.apply(calc_test_comp.positions))
            rmsd_core, _, _ = permute_align(tmp_ref_comp, calc_test_comp)
            rmsd_mirror, _, _ = mirror_align(tmp_ref_comp, calc_test_comp)
            if rmsd_mirror < rmsd_core:
                rmsd_core = rmsd_mirror
            if rmsd_core < best:
                saveq = q
                best = rmsd_core
            
        tmp_self_comp.set_positions(saveq.apply(tmp_self_comp.positions))
        genMol.ase_atoms.set_positions(saveq.apply(genMol.ase_atoms.positions))
        
        rmsd_core, r, outcore = permute_align(tmp_ref_comp, tmp_self_comp)
        rmsd_mirror, r_mirror, moutcore = mirror_align(tmp_ref_comp, tmp_self_comp)

        if rmsd_mirror < rmsd_core: # Pick the better one!
            rmsd_core = rmsd_mirror
            outcore = moutcore
            r = r_mirror
            newposits = genMol.ase_atoms.positions
            newposits[:,0] = -newposits[:,0] # Mirror across x axis to replicate mirror in permute
            genMol.ase_atoms.set_positions(newposits)
                
        rmsd_loss_core = rmsd_core
        tmp_posits = genMol.ase_atoms.positions
        tmp_posits = r.apply(tmp_posits)
        genMol.ase_atoms.set_positions(tmp_posits)
    ## Do permutation mapping to estimate full loss given the rotation to match the core.
        if compareMol.ase_atoms.get_chemical_formula() == genMol.ase_atoms.get_chemical_formula():
            rmsd_loss_full, _, _ = permute_align(copy.deepcopy(compareMol.ase_atoms), copy.deepcopy(genMol.ase_atoms), 
                                                    maxiter=1, tol=1e-6, in_place=True)
        else:
            rmsd_loss_full = None
        
    if rmsd_type == 'simple':
        # Returns per-atom RMSD for ideal/rotation/translation overlap.
        return (rmsd_loss_core, 
                rmsd_loss_full, 
                compareMol.write_mol2('refmol.mol2',writestring=True), 
                genMol.write_mol2('aligned.mol2',writestring=True),
                convert_ase_xyz(tmp_ref_comp),
                convert_ase_xyz(outcore),
                flag_struct)
    else:
        print('Not yet implemented.')
        return None