import ase
import numpy as np
from architector.io_align_mol import reorder_align
from ase.vibrations import Vibrations
from ase.constraints import FixBondLengths
from scipy.stats import bernoulli
from openbabel import openbabel
from architector.io_calc import CalcExecutor
import architector.arch_context_manage as arch_context_manage
from architector.io_molecule import convert_io_molecule
from architector.io_obabel import (convert_mol2_obmol,convert_obmol_ase)


def bond_length_sampler(relaxed_mol, n=10, seed=42, max_dev=0.4, smallest_dist_cutoff=0.55,
                        min_dist_cutoff=3,
                        final_relax=False, final_relax_steps=50, debug=False,
                        return_energies=False,
                        ase_opt_method=None,ase_opt_kwargs={}):
    """bond_length_sampler
    Attempt to sample based on bond length deviations.

    Parameters
    ----------
    relaxed_mol : architector.io_molecule.Molecule
        relaxed molcule
    n : int, optional
        number of conformers to generate, by default 10
    seed : int, optional
        random seed, by default 42
    max_dev : float, optional
        maximum deviation around bonds, by default 0.3
    smallest_dist_cutoff : float
        distance cutoff-make sure sum of cov radii larger than dist*smallest_dist_cutoff, default 0.55.
    min_dist_cutoff : int/float
        make sure all atoms are at least min_dist_cutoff from ANY other atom, default 3 angstroms
    final_relax : bool, optional
        Perform "cleaning" relxation with usually GFN2-xTB, by default False.
    final_relax_steps : int, optional
        Take n stpes before stopping. It won't converge because of constraints, but will finish.
    return_energies : bool, optional, 
        return energies and rmsds. default False
    """
    good = True
    mol2 = relaxed_mol.write_mol2('init.mol2', writestring=True)
    tmpmol = convert_io_molecule(mol2)
    relaxed_atoms = relaxed_mol.ase_atoms
    calc = relaxed_atoms.get_calculator()
    if seed:
        np.random.seed(seed)
    inda,indb = np.nonzero(relaxed_mol.graph)
    inda,indb = inda[np.where(indb > inda)], indb[np.where(indb > inda)]
    bond_dists = relaxed_atoms.get_all_distances()[(inda,indb)]
    displaced_structures = []
    energies = []
    rmsds = []
    total_out = 0
    while total_out < n:
        fail = False
        OBMol = convert_mol2_obmol(mol2)
        distortion = np.random.uniform(low=1-max_dev,
                                                     high=1+max_dev,
                                                     size=(bond_dists.shape[0]))
        new_bond_dists = bond_dists*distortion
        ase_dist_constrs = []
        constr = openbabel.OBFFConstraints()
        ff = openbabel.OBForceField.FindForceField('UFF')
        for i,a in enumerate(inda):
            a = int(a) + 1
            b = int(indb[i]) + 1
            constr.AddDistanceConstraint(a,b,float(new_bond_dists[i]))
            ase_dist_constrs.append([a-1,b-1])
        s = ff.Setup(OBMol,constr)
        if not s:
            fail=True
        try:
            for i in range(200):
                ff.SteepestDescent(10)
                ff.ConjugateGradients(10)
        except:
            fail = True
        ff.GetCoordinates(OBMol)
        if not fail:
            tmp_atoms = convert_obmol_ase(OBMol)
            out_atoms = relaxed_atoms.copy()
            out_atoms.set_positions(tmp_atoms.positions)
            tmpmol.dists_sane = True
            tmpmol.ase_atoms = out_atoms
            tmpmol.dist_sanity_checks(min_dist_cutoff=min_dist_cutoff,smallest_dist_cutoff=smallest_dist_cutoff,debug=debug)
            if final_relax:
                c = FixBondLengths(ase_dist_constrs)
                out_atoms.set_constraint(c)
            if tmpmol.dists_sane:
                out = CalcExecutor(out_atoms,method='custom',calculator=calc,relax=final_relax,
                                   maxsteps=final_relax_steps,ase_opt_method=ase_opt_method,
                                   ase_opt_kwargs=ase_opt_kwargs,debug=debug)
                if out.successful:
                    _,rmsd = reorder_align(relaxed_atoms,out.mol.ase_atoms,return_rmsd=True)
                    rmsds.append(rmsd)
                    energies.append(out.energy)
                    displaced_structures.append(out.mol)
                    total_out += 1
    if good and return_energies:
        return displaced_structures,energies,rmsds
    elif return_energies:
        return [],energies,rmsds
    elif good:
        return displaced_structures
    else:
        return []

def random_sampler(relaxed_mol, n=10, seed=42, min_rmsd=0.1, max_rmsd=0.4,
                  min_dist_cutoff=3, smallest_dist_cutoff=0.55, return_energies=False, debug=False):
    """random_sampler
    fully random sampling

    Parameters
    ----------
    relaxed_mol : architector.io_molecule.Molecule
        relaxed molecule with calculator attached to mol.ase_atoms
    temp : int, optional
        temperature for kbT approximation, by default 298.15, possibly lower later
    n : int, optional
        number of conformations to generate, by default 10
    seed : int, optional
        random seed for generation, by default 42
    smallest_dist_cutoff : float
        distance cutoff-make sure sum of cov radii larger than dist*smallest_dist_cutoff, default 0.55.
    min_dist_cutoff : int/float
        make sure all atoms are at least min_dist_cutoff from ANY other atom, default 3 angstroms
    min_rmsd : float, optional
        minimum rmsd (angstroms) for difference, default 0.1
    max_rmsd : float, optional
        maximum rmsd for difference, default 0.4
    mindist : float, optional
        minimum interatomic distance for atoms, by default 0.5
    return_energies : bool, optional, 
        return energies and rmsds. default False

    Returns
    -------
    displaced_structures : list (ase.atoms.Atoms)
        A list of the displaced structures that have been evaluated by the calculator

    Raises
    ------
    ValueError
        If the imaginary modes are quite strong
    """
    good = True
    relaxed_atoms = relaxed_mol.ase_atoms
    calc = relaxed_atoms.get_calculator()
    mol2 = relaxed_mol.write_mol2('init.mol2', writestring=True)
    tmpmol = convert_io_molecule(mol2)
    if seed:
        np.random.seed(seed)
    na = len(relaxed_atoms)
    displaced_structures = []
    energies = []
    rmsds = []
    total_out = 0
    max_dist = max_rmsd*0.9
    while total_out < n:
        out_atoms = relaxed_atoms.copy()
        # Generate random displacements
        newcoords = out_atoms.positions + np.random.uniform(low=-max_dist,
                                                     high=max_dist,
                                                     size=(na, 3))
        out_atoms.set_positions(newcoords)
        tmpmol.dists_sane = True
        tmpmol.ase_atoms = out_atoms
        _,rmsd = reorder_align(relaxed_atoms,out_atoms,return_rmsd=True)
        tmpmol.dist_sanity_checks(min_dist_cutoff=min_dist_cutoff,smallest_dist_cutoff=smallest_dist_cutoff,debug=debug)
        if (tmpmol.dists_sane) and (rmsd > min_rmsd) and (rmsd < max_rmsd):
            out = CalcExecutor(out_atoms,method='custom',calculator=calc,relax=False,debug=debug)
            if out.successful:
                rmsds.append(rmsd)
                energies.append(out.energy)
                displaced_structures.append(out.mol)
                total_out += 1
    if good and return_energies:
        return displaced_structures,energies,rmsds
    elif return_energies:
        return [],energies,rmsds
    elif good:
        return displaced_structures
    else:
        return []


def normal_mode_sampler(relaxed_mol, temp=298.15, n=10, seed=42, min_dist_cutoff=3, smallest_dist_cutoff=0.55,
                        return_energies=False, debug=False): # T=1 produced okay results in preliminary tests
    """normal_mode_sampler
    https://www.nature.com/articles/sdata2017193#Sec12

    Parameters
    ----------
    relaxed_mol : architector.io_molecule.Molecule
        relaxed molecule with calculator attached to mol.ase_atoms
    temp : int, optional
        temperature for kbT approximation, by default 298.15, possibly lower later
    n : int, optional
        number of conformations to generate, by default 10
    seed : int, optional
        random seed for generation, by default 42
    return_energies and rmsds: bool, optional, 
        return energies. default False
    smallest_dist_cutoff : float
        distance cutoff-make sure sum of cov radii larger than dist*smallest_dist_cutoff, default 0.55.
    min_dist_cutoff : int/float
        make sure all atoms are at least min_dist_cutoff from ANY other atom, default 3 angstroms

    Returns
    -------
    displaced_structures : list (ase.atoms.Atoms)
        A list of the displaced structures that have been evaluated by the calculator

    Raises
    ------
    ValueError
        If the imaginary modes are quite strong
    """
    good = True
    mol2 = relaxed_mol.write_mol2('init.mol2', writestring=True)
    relaxed_atoms = relaxed_mol.ase_atoms
    with arch_context_manage.make_temp_directory() as _:
        vib = Vibrations(relaxed_atoms)
        calc = relaxed_atoms.get_calculator()
        vib.run()
        vib_energies = vib.get_energies()
        if np.any(np.imag(vib_energies) > 0.1):
            print('Warning: There are some highly imaginary modes!')
            good = False
        vib_energies = np.real(vib_energies)
        non_zero_inds = np.where(np.real(vib_energies) > 0.0001)[0]
        kbT = ase.units.kB*temp
        # Seed
        if seed:
            np.random.seed(seed)
        na = len(relaxed_atoms)
        displaced_structures = []
        energies = []
        rmsds = []
        for i in range(n):
            out_atoms = relaxed_atoms.copy()
            # Generate random cs
            cs = np.random.random(len(non_zero_inds))
            scale = np.random.random()
            cs = cs/np.sum(cs)*scale # Set to scale between 0-1
            # Generate random signs
            signs = bernoulli.rvs(0.5,size=len(non_zero_inds))
            signs[np.where(signs == 0)] = -1
            Rs = signs*np.sqrt(3*cs*na*kbT/vib_energies[non_zero_inds])
            displaced_posits = relaxed_atoms.get_positions()
            for j,r in enumerate(Rs):
                norm_mode = vib.get_mode(non_zero_inds[j])
                displacement = norm_mode*r
                displaced_posits += displacement
            out_atoms.set_positions(displaced_posits)
            out_atoms.calc = calc
            try:
                energy = out_atoms.get_total_energy()
                tmpmol = convert_io_molecule(mol2)
                tmpmol.ase_atoms = out_atoms
                tmpmol.dist_sanity_checks(min_dist_cutoff=min_dist_cutoff,smallest_dist_cutoff=smallest_dist_cutoff,debug=debug)
                energies.append(energy)
                _,rmsd = reorder_align(relaxed_atoms,out_atoms,return_rmsd=True)
                rmsds.append(rmsd)
                displaced_structures.append(tmpmol)
            except:
                continue
    if good and return_energies:
        return displaced_structures,energies,rmsds
    elif return_energies:
        return [],energies
    elif good:
        return displaced_structures,rmsds
    else:
        return []