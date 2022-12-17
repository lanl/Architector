from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
import ase
import numpy as np
from scipy.stats import bernoulli
from architector import arch_context_manage


def calc_free_energy(relaxed_atoms ,temp=298.15, pressure=101325, geometry='nonlinear'):
    """calc_free_energy utility function to calculate free energy of relaxed structures with
    ASE calculators added.

    Uses the ideal gas rigid rotor harmonic oscillator (IGRRHO) approach

    Parameters
    ----------
    relaxed_atoms : ase.atoms.Atoms
        relaxed structures with calculator attached (usually XTB)
    temp : float, optional
        temperature in kelvin, by default 298.15
    pressure : int, optional
        pressure in pascal, by default 101325 Pa
    geometry : str, optional
        'linear','nonlinear' , by default 'nonlinear'

    Returns
    -------
    G, float
       free energy in eV
    thermo, ase.thermo
        ASE thermo calculator
    """
    with arch_context_manage.make_temp_directory() as _:
        potentialenergy = relaxed_atoms.get_potential_energy()
        vib = Vibrations(relaxed_atoms)
        vib.run()
        vib_energies = vib.get_energies()
        nunpaired = np.sum(relaxed_atoms.get_initial_magnetic_moments())

        thermo = IdealGasThermo(vib_energies=vib_energies,
                                potentialenergy=potentialenergy,
                                atoms=relaxed_atoms,
                                geometry=geometry,
                                symmetrynumber=2, spin=nunpaired/2)
        G = thermo.get_gibbs_energy(temperature=temp, pressure=pressure)
    return G, thermo


def normal_mode_sample(relaxed_atoms,temp=298.15,n=10,seed=42): # T=1 produced okay results in preliminary tests
    """normal_mode_sample
    https://www.nature.com/articles/sdata2017193#Sec12

    Parameters
    ----------
    relaxed_atoms : ase.atoms.Atoms
        relaxed atoms with calculator attached
    temp : int, optional
        temperature for kbT approximation, by default 298.15, possibly lower later
    n : int, optional
        number of conformations to generate, by default 10
    seed : int, optional
        random seed for generation, by default 42

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
                out_atoms.get_total_energy()
                displaced_structures.append(out_atoms)
            except:
                continue
    if good:
        return displaced_structures
    else:
        return []