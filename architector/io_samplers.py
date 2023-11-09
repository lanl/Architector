import ase
import numpy as np
import os.path as osp
from architector.io_align_mol import (reorder_align_rmsd, simple_rmsd)
from ase.vibrations import Vibrations
from ase.constraints import FixBondLengths
from scipy.stats import bernoulli
from openbabel import openbabel
from tqdm import tqdm
from architector.io_calc import CalcExecutor
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
import architector.arch_context_manage as arch_context_manage
from architector.io_molecule import convert_io_molecule
from architector.vibrations_free_energy import vibration_analysis
from architector.io_obabel import (convert_mol2_obmol, convert_obmol_ase)

def md_sampler(relaxed_mol, temp=298.15, interval=20, n=50, warm_up=1000, 
               timestep=1.0, friction=0.02, return_energies=False, debug=False):
    """md_sampler
    Use langevin dynamics and specified temperature to sample structures.

    Parameters
    ----------
    relaxed_mol : architector.io_molecule.Molecule
        relaxed molcule
    temp : float, optional
        temperature, by default 298.15
    interval : int, optional
        time interval to save sampled structures, by default 50
    n : int, optional
        Number of samples to save, by default 50
    warm_up : int, optional
       Warm-up interval from 0K, by default 1000
    timestep : float, optional
        timestep in fs, by default 1.0
    friction : float, optional
        langevin friction, by default 0.02
    debug : bool, optional
        Print out stuff, by default False
    return_energies : bool, optional
       give back energies and rmsds, by default False
    """
    # Friction increased to get to convergence faster.
    mol2 = relaxed_mol.write_mol2('init.mol2', writestring=True)
    init_ase = convert_io_molecule(mol2).ase_atoms
    relaxed_atoms = relaxed_mol.ase_atoms
    skip_n = int(warm_up/interval) 
    good = True
    with tqdm(total=n+int(warm_up/interval)) as pbar:
        with arch_context_manage.make_temp_directory() as tdir:
            dyn = Langevin(relaxed_atoms, timestep * ase.units.fs, temp* ase.units.kB, friction=friction)
            def printenergy(a=relaxed_atoms):  # store a reference to atoms in the definition.
                """Function to print the potential, kinetic and total energy."""
                epot = a.get_potential_energy() / len(a)
                ekin = a.get_kinetic_energy() / len(a)
                print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                    'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * ase.units.kB), epot + ekin))
            def incremental(a=relaxed_atoms):
                pbar.update(1)
            if debug:
                dyn.attach(printenergy, interval=interval)
            traj = Trajectory('moldyn3.traj', 'w', relaxed_atoms)
            dyn.attach(traj.write, interval=interval)
            dyn.attach(incremental,interval=interval)
            # Now run the dynamics
            dyn.run(warm_up + n*interval)
            traj = Trajectory(osp.join(tdir,'moldyn3.traj'))
            trunc_traj = traj[skip_n:]
            displaced_structures = []
            energies = []
            full_results = []
            simple_rmsds = []
            aligned_rmsds = []
            for image in trunc_traj:
                tmpmol = convert_io_molecule(mol2)
                s_rmsd = simple_rmsd(init_ase,image)
                _,align_rmsd = reorder_align_rmsd(init_ase,image,return_rmsd=True)
                energies.append(image.calc.results['energy'])
                full_results.append(image.calc.results)
                tmpmol.ase_atoms = image
                displaced_structures.append(tmpmol)
                simple_rmsds.append(s_rmsd)
                aligned_rmsds.append(align_rmsd)
    if good and return_energies:
        return displaced_structures, energies, full_results, simple_rmsds, aligned_rmsds
    elif return_energies:
        return [], energies, full_results, simple_rmsds, aligned_rmsds
    elif good:
        return displaced_structures, simple_rmsds, aligned_rmsds
    else:
        return []


def bond_length_sampler(relaxed_mol,
                        n=10,
                        seed=42,
                        max_dev_low=0.1,
                        max_dev_hi=0.3,
                        smallest_dist_cutoff=0.55,
                        min_dist_cutoff=3,
                        final_relax=False,
                        final_relax_steps=50,
                        ase_opt_method=None,
                        ase_opt_kwargs={},
                        max_attempts=10000,
                        return_energies=False,
                        debug=False):
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
    max_dev_low : float, optional
        maximum compression of bonds, by default 0.1
    max_dev_hi : float, optional
        maximum stretch of bonds, by default 0.4
    smallest_dist_cutoff : float
        distance cutoff-make sure sum of cov radii larger than dist*smallest_dist_cutoff, default 0.55.
    min_dist_cutoff : int/float
        make sure all atoms are at least min_dist_cutoff from ANY other atom, default 3 angstroms
    final_relax : bool, optional
        Perform "cleaning" relxation with usually GFN2-xTB, by default False.
    final_relax_steps : int, optional
        Take n stpes before stopping. It won't converge because of constraints, but will finish.
    return_energies : bool, optional 
        return energies and rmsds. default False
    ase_opt_method : None, optional 
        ASE optimizer class used for geometry optimizations. Default will use LBFGSLineSearch.
    ase_opt_kwargs : dict(), 
        ASE optimizer kwargs. Do not include "trajectory" nor "logfile" kwargs.
    max_attempts : int, optional
        maximum possible number of attempts beyond n, by default 1000
    debug : bool, optional
        print debugging statements, by default False
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
    max_attempts = n + max_attempts
    displaced_structures = []
    energies = []
    full_results = []
    simple_rmsds = []
    aligned_rmsds = []
    total_out = 0
    count = 0
    with tqdm(total=n) as pbar:
        while (total_out < n) and (count < max_attempts):
            count += 1
            fail = False
            OBMol = convert_mol2_obmol(mol2)
            distortion = np.random.uniform(low=1-max_dev_low,
                                        high=1+max_dev_hi,
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
                if tmpmol.dists_sane and return_energies:
                    out = CalcExecutor(out_atoms,method='custom',calculator=calc,relax=final_relax,
                                    maxsteps=final_relax_steps,ase_opt_method=ase_opt_method,
                                    ase_opt_kwargs=ase_opt_kwargs,debug=debug)
                    if out.successful:
                        s_rmsd = simple_rmsd(relaxed_atoms,out.mol.ase_atoms)
                        _,align_rmsd = reorder_align_rmsd(relaxed_atoms,
                                                          out.mol.ase_atoms,
                                                          return_rmsd=True)
                        aligned_rmsds.append(align_rmsd)
                        simple_rmsds.append(s_rmsd)
                        energies.append(out.energy)
                        full_results.append(out.mol.ase_atoms.calc.results)
                        tmpmol = convert_io_molecule(mol2)
                        tmpmol.ase_atoms = out.mol.ase_atoms
                        displaced_structures.append(tmpmol)
                        total_out += 1
                        pbar.update(1)
                elif tmpmol.dists_sane:
                    s_rmsd = simple_rmsd(relaxed_atoms,tmpmol.ase_atoms)
                    _,align_rmsd = reorder_align_rmsd(relaxed_atoms,tmpmol.ase_atoms,return_rmsd=True)
                    simple_rmsds.append(s_rmsd)
                    aligned_rmsds.append(align_rmsd)
                    displaced_structures.append(tmpmol)
                    total_out += 1
                    pbar.update(1)
    if good and return_energies:
        return displaced_structures,energies,full_results,simple_rmsds,aligned_rmsds
    elif return_energies:
        return [],energies,full_results,simple_rmsds,aligned_rmsds
    elif good:
        return displaced_structures,simple_rmsds,aligned_rmsds
    else:
        return []

def random_sampler(relaxed_mol, n=10, seed=42, min_rmsd=0.1, max_rmsd=0.5,
                   min_dist_cutoff=3, smallest_dist_cutoff=0.55, 
                   return_energies=False, max_attempts=10000,
                   debug=False):
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
        maximum rmsd for difference, default 0.5
    mindist : float, optional
        minimum interatomic distance for atoms, by default 0.5
    return_energies : bool, optional, 
        return energies and rmsds. default False
    max_attempts : int, optional
        maximum possible number of attempts beyond n, by default 10000
    debug : bool, optional
        print debugging statements, by default False

    Returns
    -------
    displaced_structures : list (ase.atoms.Atoms)
        A list of the displaced structures that have been evaluated by the calculator
    """
    good = True
    relaxed_atoms = relaxed_mol.ase_atoms
    calc = relaxed_atoms.get_calculator()
    mol2 = relaxed_mol.write_mol2('init.mol2', writestring=True)
    if seed:
        np.random.seed(seed)
    na = len(relaxed_atoms)
    displaced_structures = []
    energies = []
    full_results = []
    simple_rmsds = []
    aligned_rmsds = []
    total_out = 0
    max_attempts = max_attempts + n
    count = 0
    max_dist = max_rmsd*0.9
    with tqdm(total=n) as pbar:
        while (total_out < n) and (count < max_attempts):
            count += 1
            out_atoms = relaxed_atoms.copy()
            # Generate random displacements
            newcoords = out_atoms.positions + np.random.uniform(low=-max_dist,
                                                        high=max_dist,
                                                        size=(na, 3))
            out_atoms.set_positions(newcoords)
            tmpmol = convert_io_molecule(mol2)
            tmpmol.dists_sane = True
            s_rmsd = simple_rmsd(relaxed_atoms,out_atoms)
            _,align_rmsd = reorder_align_rmsd(relaxed_atoms,out_atoms,return_rmsd=True)
            tmpmol.ase_atoms = out_atoms
            tmpmol.dist_sanity_checks(min_dist_cutoff=min_dist_cutoff,smallest_dist_cutoff=smallest_dist_cutoff,debug=debug)
            if (tmpmol.dists_sane) and (s_rmsd > min_rmsd) and (s_rmsd < max_rmsd) and (return_energies):
                out = CalcExecutor(out_atoms,method='custom',calculator=calc,relax=False,debug=debug)
                if out.successful:
                    simple_rmsds.append(s_rmsd)
                    aligned_rmsds.append(align_rmsd)
                    energies.append(out.energy)
                    full_results.append(out.ase_atoms.calc.results)
                    displaced_structures.append(tmpmol)
                    total_out += 1
                    pbar.update(1)
            elif (tmpmol.dists_sane) and (s_rmsd > min_rmsd) and (s_rmsd < max_rmsd):
                simple_rmsds.append(s_rmsd)
                aligned_rmsds.append(align_rmsd)
                displaced_structures.append(tmpmol)
                total_out += 1
                pbar.update(1)
    if count == max_attempts:
        good = False
    if good and return_energies:
        return displaced_structures,energies,full_results,simple_rmsds,aligned_rmsds
    elif return_energies:
        return [],energies,full_results,simple_rmsds,aligned_rmsds
    elif good:
        return displaced_structures,simple_rmsds,aligned_rmsds
    else:
        return []


def normal_mode_sampler(relaxed_mol, 
                        hess=None,
                        temp=298.15, 
                        n=10, 
                        seed=42,
                        distance_factor=1.0,
                        freq_cutoff=150,
                        mode_type='mass_weighted_unnormalized',
                        linear=False,
                        n_modes_to_sample=None,
                        per_mode_temp=False,
                        min_dist_cutoff=3,
                        smallest_dist_cutoff=0.55,
                        return_energies=False, 
                        debug=False):
    """normal_mode_sampler
    https://www.nature.com/articles/sdata2017193#Sec12


    Parameters
    ----------
    relaxed_mol : architector.io_molecule.Molecule
        relaxed molecule with ASE calculator attached to mol.ase_atoms
    hess : np.ndarray/None, optional
        2D Hessian in eV/Angstroms^2 either generated by ase.vibrations.Vibrations or from an external program.
        e.g.: 
            hess = np.array([[at1x_at1x, at1x_at1y, at1x_at1z, at1x_at2x, ...],
                            [at1y_at1x, at1y_at1y, at1y_at1z, at1y_at2x, ...],
                            [at1z_at1x, at1z_at1y, at1z_at1z, at1z_at2x, ...],
                            [at2x_at1x, at2x_at1y, at2x_at1z, at2x_at2x, ...],
                            ...])
    temp : int, optional
        temperature for kbT approximation, by default 298.15, possibly lower later
    n : int, optional
        number of conformations to generate, by default 10
    seed : int, optional
        random seed for generation, by default 42
    distance_factor : float, optional
        Scale normal mode displacements uniformly by factor to approach "correct" temperature. by default 1.0 (no scaling)
        This may be related to modes canceling each other out when summed usually resulting in a slight to major underestimation of the "temperature"
        of the displaced
    freq_cutoff : float, optional
        Modes with frequency below this value (cm^-1) will not be sampled. Having a cutoff makes a huge difference for metal-organics.
        without the cutoff the low-frequency (low-energy) modes can dominate displacements resulting in VERY artificially high "temperature" samples.
        By defualt 150 cm^-1 from tests on Ni/Fe aqua complexes with XTB. Cutoff imposed did not affect energies of organic molecules sampled.
    mode_type : str, optional
        What type of modes do you want to apply the sampling to. By default 'mass_weight_unnormalized' since this gives most accurate temperatures 
        across molecule types.
    linear : bool, optional
        Is this a linear molecule?, default False
    n_modes_to_sample : None/int, optional
        Number of modes to sample across starting from lowest-energy modes if specified as int. by default None (aka Across all "good" modes)
    per_mode_temp : bool, optional
        Whether to normalize the temperature/energy per-mode. default False.
    return_energies and rmsds: bool, optional, 
        return energies. default False
    smallest_dist_cutoff : float
        distance cutoff-make sure sum of cov radii larger than dist*smallest_dist_cutoff, default 0.55.
    min_dist_cutoff : int/float
        make sure all atoms are at least min_dist_cutoff from ANY other atom, default 3 angstroms
    return_energies : bool, optional, 
        return energies and rmsds. default False
    debug : bool, optional
        print debugging statements, by default False

    Returns
    -------
    displaced_structures : list (ase.atoms.Atoms)
        A list of the displaced structures that have been evaluated by the calculator

    Raises
    ------
    Warning - If the imaginary modes are quite strong
    """
    good = True
    mol2 = relaxed_mol.write_mol2('init.mol2', writestring=True)
    relaxed_atoms = relaxed_mol.ase_atoms
    with arch_context_manage.make_temp_directory() as _:
        if (hess is None):
            vib_analysis = Vibrations(relaxed_atoms)
            vib_analysis.run()
            data = vib_analysis.get_vibrations()
            hess = data.get_hessian_2d()
        vib_energies, modes, fconstants, _ , frequencies = vibration_analysis(relaxed_atoms,
                                                                              hess,
                                                                              mode_type=mode_type)
    
        # _ is rmasses matrix in case it is needed later.
        if linear: # 3N-5
            vib_energies = np.real(vib_energies[5:])
            modes = modes[5:]
            fconstants = np.real(fconstants[5:])
            frequencies = np.real(frequencies[5:])
        else: # 3N-6
            vib_energies = np.real(vib_energies[6:])
            modes = modes[6:]
            fconstants = np.real(fconstants[6:])
            frequencies = np.real(frequencies[6:])
        if n_modes_to_sample is not None:
            vib_energies = vib_energies[:n_modes_to_sample]
            modes = modes[:n_modes_to_sample]
            fconstants = fconstants[:n_modes_to_sample]
            frequencies = frequencies[:n_modes_to_sample]
        if np.any(np.imag(vib_energies) > 0.1):
            print('Warning: There are some highly imaginary modes!')
            good = False
        if debug:
            print('Vib Energies',vib_energies)
        non_zero_inds = np.where(frequencies > freq_cutoff)[0]
        nmodes = len(non_zero_inds)
        kbT = ase.units.kB*temp
        if seed:
            np.random.seed(seed)
        displaced_structures = []
        energies = []
        full_results = []
        # estimated_energies = [] # Used for debugging sampling method.
        simple_rmsds = []
        aligned_rmsds = []
        calc = relaxed_atoms.get_calculator()
        for _ in tqdm(range(n),total=n):
            out_atoms = relaxed_atoms.copy()
            # Generate random cs
            cs = np.random.random(nmodes)
            # Generate random signs
            signs = bernoulli.rvs(0.5,size=nmodes)
            signs[np.where(signs == 0)] = -1
            # Average energy contribution per-mode
            if per_mode_temp:
                per_mode_e = 2*kbT/nmodes
            else:
                per_mode_e = 2*kbT
            Rs = signs*np.sqrt(per_mode_e/fconstants[non_zero_inds]*np.log(1/(1-cs))) 
            # estimated_energy_vect = [1/2*f*Rs[i]**2 for i,f in enumerate(fconstants[non_zero_inds])]
            Rs = Rs*distance_factor # Fudge factor for specific methods based on experience for GFn2-xTB - 1 works well.
            displaced_posits = relaxed_atoms.get_positions()
            for j,r in enumerate(Rs):
                norm_mode = modes[non_zero_inds[j]]
                displacement = norm_mode*r
                displaced_posits += displacement
            out_atoms.set_positions(displaced_posits)
            try:
                out_atoms.calc = calc
                if return_energies: # Only perform energy evaluation if requested.
                    energy = out_atoms.get_total_energy()
                    energies.append(energy)
                    full_results.append(out_atoms.calc.results)
                tmpmol = convert_io_molecule(mol2)
                s_rmsd = simple_rmsd(relaxed_atoms,out_atoms)
                _,aligned_rmsd = reorder_align_rmsd(relaxed_atoms,out_atoms,return_rmsd=True)
                tmpmol.ase_atoms = out_atoms
                tmpmol.dist_sanity_checks(min_dist_cutoff=min_dist_cutoff,
                                          smallest_dist_cutoff=smallest_dist_cutoff,
                                          debug=debug)
                simple_rmsds.append(s_rmsd)
                aligned_rmsds.append(aligned_rmsd)
                displaced_structures.append(tmpmol)
            except:
                continue
    if good and return_energies:
        return displaced_structures,energies, full_results, simple_rmsds, aligned_rmsds, hess
    elif return_energies:
        return [], energies, full_results
    elif good:
        return displaced_structures, simple_rmsds, aligned_rmsds, hess
    else:
        return []