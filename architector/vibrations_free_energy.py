from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
import numpy as np
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