"""
Routines for performing ASE NEB from two structures generated
at different distances

Developed by Michael Taylor
"""

from ase.neb import NEB
from ase.constraints import Hookean
from architector.io_align_mol import reorder_align_rmsd
from architector.io_calc import CalcExecutor
from architector.io_molecule import convert_io_molecule
import architector.io_ptable as io_ptable
from scipy.optimize import linear_sum_assignment
import numpy as np
import copy


def permutation_cost_mat_neb(atoms1, atoms2,
                             nimages=8,
                             neb_method='UFF',
                             interpolation='idpp',
                             skin=0.2,
                             cost_type='default_min_emax'):
    """permutation_cost_mat_neb
    return the cost matrix required by Hungarian method
    Only permutation with same label is allowed

    Parameters
    -------
    costtype : str, optional
        what type of cost/distance loss to use, by default 'xyz'
        'xyz' - pairwise distance from point of 1 to point in 2
        'COM' - distance to the center of mass

    Returns
    -------
    cost_mat : np.ndarray
        2D array NXN containing costs for the hungarian method
    """
    npt = len(atoms1)
    images = [atoms1]
    images += [atoms1.copy() for _ in range(nimages-2)]
    images += [atoms2]
    neb = NEB(images=images)
    neb.interpolate(method=interpolation)
    mol0 = convert_io_molecule(atoms1)
    mol0.create_mol_graph()
    bond_change_inds = []
    for image in neb.images[1:]:
        tmol = convert_io_molecule(image)
        tmol.create_mol_graph(skin=skin)
        bond_change_inds += np.nonzero(tmol.graph - mol0.graph)[0].tolist()
    bond_change_inds = sorted(list(set(bond_change_inds)))
    print(bond_change_inds)
    cost_mat = np.zeros([npt, npt])
    label1_t = atoms1.get_atomic_numbers()
    for x, i in enumerate(bond_change_inds[:-1]):
        for j in bond_change_inds[x+1:]:
            if label1_t[i] != label1_t[j]:
                # permutation between different element is not allowed
                cost_mat[i, j] = np.inf
                cost_mat[j, i] = np.inf
            else:
                tmp1 = atoms1.copy()
                tmp2 = atoms2.copy()
                reorder = np.arange(len(tmp1))
                reorder[i] = j
                reorder[j] = i
                tmp1 = tmp1[reorder]
                images = [tmp1]
                images += [tmp1.copy() for _ in range(nimages-2)]
                images += [tmp2]
                neb = NEB(images=images)
                neb.interpolate(method=interpolation)
                energies = []
                for img in neb.images:
                    tmol = convert_io_molecule(img)
                    tout = CalcExecutor(tmol, method=neb_method)
                    energies.append(tout.energy)
                cost_mat[i, j] = max(energies)
                cost_mat[j, i] = max(energies)
    zeroinds = np.where(cost_mat == 0)
    cost_mat[zeroinds] = np.inf
    return cost_mat


def check_bonds(mol,
                bonds_breaking,
                bonds_forming,
                breaking_cutoff,
                forming_cutoff):
    dists = mol.ase_atoms.get_all_distances()
    anums = mol.ase_atoms.get_atomic_numbers()
    goods = []
    for inds in bonds_breaking:
        cutoff_dist = (io_ptable.rcov1[anums[inds[0]]] + io_ptable.rcov1[anums[inds[1]]])*breaking_cutoff
        actual_dist = dists[inds[0]][inds[1]]
        if actual_dist > cutoff_dist:
            goods.append(True)
        else:
            goods.append(False)
    for inds in bonds_forming:
        cutoff_dist = (io_ptable.rcov1[anums[inds[0]]] + io_ptable.rcov1[anums[inds[1]]])*forming_cutoff
        actual_dist = dists[inds[0]][inds[1]]
        if actual_dist < cutoff_dist:
            goods.append(True)
        else:
            goods.append(False)
    return np.all(goods)


def qm_neb(initial,
           final,
        # nimages=8, # Prune/expand trajectory - possibly implement if needed.
           breaking_cutoff=1.5,
           forming_cutoff=1.2,
           start_force_constant=0.05,
           force_increment=0.1,
           structure_match_fconst=0.001,
           method='GFN2-xTB',
           max_steps=4,
           fmax_opt=0.1,
           ):
    """_summary_

    Parameters
    ----------
    initial : _type_
        _description_
    final : _type_
        _description_
    nimages : int, optional
        _description_, by default 8
    forming_cutoff : float, optional
        _description_, by default 1.2
    start_force_constant : float, optional
        _description_, by default 0.05
    force_increment : float, optional
        _description_, by default 0.5
    structure_match_fconst : float, optional
        _description_, by default 0.2
    method : str, optional
        _description_, by default 'GFN2-xTB'
    fmax_opt : float, optional
        _description_, by default 0.1

    Returns
    -------
    _type_
        _description_
    """
    keep_going = True
    mol1 = convert_io_molecule(initial)
    mol2 = convert_io_molecule(final)
    mol1.create_mol_graph()
    mol2.create_mol_graph()
    bonds_forming = [(int(x[0]), int(x[1])) for x in zip(*np.where((mol2.graph - mol1.graph) == 1)) if x[0] < x[1]]
    bonds_breaking = [(int(x[0]), int(x[1])) for x in zip(*np.where((mol2.graph - mol1.graph) == -1)) if x[0] < x[1]]
    fconst = start_force_constant
    save_trajectory = []
    opt_mol = copy.deepcopy(mol1)
    while keep_going:
        print('Running Fconst = {}'.format(fconst))
        opt_mol.ase_atoms.set_constraint()
        constraints = []
        for inds in bonds_forming:
            constraint = Hookean(inds[0], inds[1], k=fconst, rt=1)
            constraints.append(constraint)
        for inds in bonds_breaking:
            constraint = Hookean(inds[0], inds[1], k=-fconst, rt=1)
            constraints.append(constraint)
        for ind in range(mol1.graph.shape[0]):
            constraint = Hookean(ind, mol2.ase_atoms.positions[ind],
                                 k=structure_match_fconst,
                                 rt=0.1)
            constraints.append(constraint)
        opt_mol.ase_atoms.set_constraint(constraints)
        tmpopt = CalcExecutor(opt_mol,
                              method=method,
                              relax=True,
                              fmax=fmax_opt,
                              maxsteps=max_steps,
                              use_constraints=True)
        tmpopt.mol.ase_atoms.calc = None
        save_trajectory.append(copy.deepcopy(tmpopt.mol))
        opt_mol = tmpopt.mol
        good = check_bonds(opt_mol, bonds_breaking, bonds_forming,
                           breaking_cutoff, forming_cutoff)
        if good:
            keep_going = False
        else:
            fconst += force_increment
    return save_trajectory


def NEB_setup(initial,
              final,
              nimages=8,
              neb_method='GFN2-xTB',
              interpolation='idpp',
              reorder_fancy=False,
              climb=True):
    initial_mol = convert_io_molecule(initial)
    charges = initial_mol.ase_atoms.get_initial_charges()
    magmoms = initial_mol.ase_atoms.get_initial_magnetic_moments()
    if 'xtb' in neb_method.lower():
        charges[0] = initial_mol.xtb_charge
        magmoms[0] = initial_mol.xtb_uhf
    else:
        charges[0] = initial_mol.charge
        magmoms[0] = initial_mol.uhf
    final_mol = convert_io_molecule(final)
    final_mol_ase = reorder_align_rmsd(initial_mol.ase_atoms,
                                       final_mol.ase_atoms)
    final_mol.ase_atoms = final_mol_ase
    if reorder_fancy:
        costmat = permutation_cost_mat_neb(initial_mol.ase_atoms, 
                                           final_mol.ase_atoms,
                                           nimages=nimages,
                                           neb_method=neb_method,
                                           interpolation=interpolation,
                                           cost_type='default_min_emax')
        permute = linear_sum_assignment(costmat)[1]
        initial_mol.ase_atoms = initial_mol.ase_atoms[permute]
    images = [initial_mol.ase_atoms]
    images += [initial_mol.ase_atoms.copy() for _ in range(nimages-2)]
    images += [final_mol.ase_atoms]
    neb = NEB(images=images,
              climb=climb)
    neb.interpolate(method=interpolation)
    neb_images = []
    for image in neb.images:
        image.set_initial_magnetic_moments(magmoms)
        image.set_initial_charges(charges)
        out = CalcExecutor(image,
                           method=neb_method)
        neb_images.append(out.mol.ase_atoms)
    neb.images = neb_images
    return neb
