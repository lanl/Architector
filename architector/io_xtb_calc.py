# """
# Assign XTB calculator to ase_atoms object following defualts or user input!
# Written by Michael Taylor

# This module is largely depricated in architector in favor of io_calc.py
# It is useful still for xtb evalulations away from architector's main codebase.
# """

import numpy as np

from xtb.ase.calculator import XTB

from architector import arch_context_manage, io_molecule
import architector.io_ptable as io_ptable

import ase
from ase.optimize import BFGSLineSearch
from ase.atoms import Atom,Atoms

def set_XTB_calc(ase_atoms, parameters=dict(), assembly=False, isCp_lig=False):
    """set_XTB_calc 
    assign xtb calculator to atoms instance!

    Parameters
    ----------
    ase_atoms : ase.atoms.Atoms
        atoms to assign calculator to
    parameters : dict, optional
        parameters from input/ io_process_input!
    assembly : bool, optional
        whether or not this is assembly or final relaxation, by default False
    isCP_lig : bool, optional
        whether this is cP ligand evaluation or not, by default False
    """
    if isCp_lig:
        ase_atoms.set_initial_charges(np.zeros(len(ase_atoms)))
        ase_atoms.set_initial_magnetic_moments(np.zeros(len(ase_atoms)))
        calc = XTB(method="GFN-FF") # Defaul to only GFN-FF for ligand conformer relaxation.
    else:
        # Charge -> charges already assigned to components during assembly
        if (parameters['full_charge'] is not None) and (not assembly):
            charge_vect = np.zeros(len(ase_atoms))
            charge_vect[0] = parameters['full_charge']
            ase_atoms.set_initial_charges(charge_vect)
        else:
            charge_vect = ase_atoms.get_initial_charges()

        mol_charge = np.sum(charge_vect)
        symbols = ase_atoms.get_chemical_symbols()
        metals = [x for x in symbols if x in io_ptable.all_metals]

        f_in_core = False
        
        if len(metals) == 1:
            if metals[0] in io_ptable.heavy_metals:
                f_in_core = True
        else:
            print('No metals - continuing anyway with obviously no f in core elements.')

        # Handle spin / magnetism
        even_odd_electrons = (np.sum([atom.number for atom in ase_atoms])-mol_charge) % 2
        if (parameters['full_spin'] is not None) and (not assembly):
            uhf = parameters['full_spin']
            uhf_start = np.zeros(len(ase_atoms))
            uhf_start[0] = uhf
            ase_atoms.set_initial_magnetic_moments(uhf_start)
        else:
            uhf = parameters['metal_spin'] # Metal spin set by io_process_input to defaults.
            if (even_odd_electrons == 1) and (uhf == 0):
                uhf = 1
            elif (even_odd_electrons == 1) and (uhf < 7) and (uhf % 2 == 0):
                uhf += 1
            elif (even_odd_electrons == 1) and (uhf >= 7) and (uhf % 2 == 0):
                uhf -= 1
            if (even_odd_electrons == 0) and (uhf % 2 == 1):
                uhf = uhf - 1 
            elif (even_odd_electrons == 1) and (uhf % 2 == 0):
                uhf = uhf + 1
            uhf_start = np.zeros(len(ase_atoms))
            if not f_in_core:
                uhf_start[0] = uhf
            else: # F in core assumes for a 3+ lanthanide there are 11 valence electrons (8 once the 3+ is taken into account)
                even_odd_electrons = (np.sum([atom.number for atom in ase_atoms]))
                even_odd_electrons = even_odd_electrons - io_ptable.elements.index(metals[0]) + 11 - mol_charge
                even_odd_electrons = even_odd_electrons % 2
                if (even_odd_electrons == 0):
                    uhf_start[0] = 0
                else:
                    uhf_start[0] = 1
            ase_atoms.set_initial_magnetic_moments(uhf_start)

        if assembly:
            if parameters['assemble_method'] == 'GFN-FF': # Need to turn off charges for GFN-FF evaluation. Probably an XTB-end bug.
                ase_atoms.set_initial_charges(np.zeros(len(ase_atoms)))
                ase_atoms.set_initial_magnetic_moments(np.zeros(len(ase_atoms)))
            calc = XTB(method=parameters['assemble_method'], solvent=parameters['solvent'])
        else:
            if parameters['full_method'] == 'GFN-FF': # Need to turn off charges for GFN-FF evaluation. Probably an XTB-end bug.
                ase_atoms.set_initial_charges(np.zeros(len(ase_atoms)))
                ase_atoms.set_initial_magnetic_moments(np.zeros(len(ase_atoms)))
            calc = XTB(method=parameters['full_method'], solvent=parameters['solvent'])

    #########################################################
    ########### Calculator Now Set! #########################
    #########################################################
    ase_atoms.calc = calc
    return ase_atoms

def set_XTB_calc_lig(ase_atoms, charge=None, uhf=None, method='GFN2-xTB',solvent='none'):
    """set_XTB_calc 
    assign xtb calculator to ligand atoms instance!

    Parameters
    ----------
    ase_atoms : ase.atoms.Atoms
        atoms to assign calculator to
    charge : int, optional
        charge of the species, default to initial charges set on ase_atoms
    uhf : int, optional
        number of unpaired electrons in the system, default to 0
    method : str, optional
        which gfn family method to use, default GFN2-xTB
    solvent: str, optional
        use a solvent?, default 'none'
    """
    if charge:
        mol_charge = charge
    else:
        mol_charge = np.sum(ase_atoms.get_initial_charges())

    charge_vect = np.zeros(len(ase_atoms))
    charge_vect[0] = mol_charge
    ase_atoms.set_initial_charges(charge_vect)
        
    # Handle spin / magnetism
    even_odd_electrons = (np.sum([atom.number for atom in ase_atoms])-mol_charge) % 2
    if (uhf is not None):
        uhf = uhf
        if (even_odd_electrons == 1) and (uhf == 0):
            uhf = 1
        elif (even_odd_electrons == 1) and (uhf < 7) and (uhf % 2 == 0):
            uhf += 1
        elif (even_odd_electrons == 1) and (uhf >= 7) and (uhf % 2 == 0):
            uhf -= 1
        if (even_odd_electrons == 0) and (uhf % 2 == 1):
            uhf = uhf - 1 
        elif (even_odd_electrons == 1) and (uhf % 2 == 0):
            uhf = uhf + 1
        uhf_start = np.zeros(len(ase_atoms))
        uhf_start[0] = uhf
        ase_atoms.set_initial_magnetic_moments(uhf_start)
    else:
        uhf = 0 # Set spin to LS by default
        if (even_odd_electrons == 1) and (uhf == 0):
            uhf = 1
        elif (even_odd_electrons == 1) and (uhf < 7) and (uhf % 2 == 0):
            uhf += 1
        elif (even_odd_electrons == 1) and (uhf >= 7) and (uhf % 2 == 0):
            uhf -= 1
        if (even_odd_electrons == 0) and (uhf % 2 == 1):
            uhf = uhf - 1 
        elif (even_odd_electrons == 1) and (uhf % 2 == 0):
            uhf = uhf + 1
        uhf_start = np.zeros(len(ase_atoms))
        uhf_start[0] = uhf
        ase_atoms.set_initial_magnetic_moments(uhf_start)

    if method == 'GFN-FF': # Need to turn off charges for GFN-FF evaluation. Probably an XTB-end bug.
        ase_atoms.set_initial_charges(np.zeros(len(ase_atoms)))
        ase_atoms.set_initial_magnetic_moments(np.zeros(len(ase_atoms)))
    calc = XTB(method=method, solvent=solvent)

    #########################################################
    ########### Calculator Now Set! #########################
    #########################################################
    ase_atoms.calc = calc

def set_XTB_calc_straight(ase_atoms, charge=None, uhf=None, method='GFN2-xTB',solvent='none'):
    """set_XTB_calc_straight
    assign xtb calculator atoms object with exaclty spin/charge requeste

    Parameters
    ----------
    ase_atoms : ase.atoms.Atoms
        atoms to assign calculator to
    charge : int, optional
        charge of the species, default to initial charges set on ase_atoms
    uhf : int, optional
        number of unpaired electrons in the system, default to 0
    method : str, optional
        which gfn family method to use, default GFN2-xTB
    solvent: str, optional
        use a solvent?, default 'none'
    """
    if charge:
        mol_charge = charge
    else:
        mol_charge = np.sum(ase_atoms.get_initial_charges())

    charge_vect = np.zeros(len(ase_atoms))
    charge_vect[0] = mol_charge
    ase_atoms.set_initial_charges(charge_vect)
        
    uhf_start = np.zeros(len(ase_atoms))
    uhf_start[0] = uhf
    ase_atoms.set_initial_magnetic_moments(uhf_start)

    if method == 'GFN-FF': # Need to turn off charges for GFN-FF evaluation. Probably an XTB-end bug.
        ase_atoms.set_initial_charges(np.zeros(len(ase_atoms)))
        ase_atoms.set_initial_magnetic_moments(np.zeros(len(ase_atoms)))
    calc = XTB(method=method, solvent=solvent)

    #########################################################
    ########### Calculator Now Set! #########################
    #########################################################
    ase_atoms.calc = calc


def xtb_relax(structure, charge=None, uhf=None, method='GFN2-xTB',solvent='none',fmax=0.05,
            detect_charge_spin=False):
    """xtb_relax relax the structure with xtb

    Parameters
    ----------
    structure : any 3D structure
       xyz, mol2string, ase atoms ...
    charge :int, optional
        total charge on the system, by default None
    uhf : int, optional
        number of unpaired electrons in the system, by default None
    method : str, optional
        which method to use, by default 'GFN2-xTB'
    solvent : str, optional
        any name xtb solvent, by default 'none'
    fmax : float, optional
        default 0.05 eV/Angstrom
    detect_charge_spin : bool, optional
        Use obmol and io_ptable metal defaults to assign charges and spins?, default False.

    Returns
    -------
    ase_atoms : ase.atoms.Atoms
        relaxed structure
    good : bool
        whether the relaxation was succesful!
    """
    if isinstance(structure,ase.atoms.Atoms):
        ase_atoms = structure
        if ase_atoms.calc is None:
            set_XTB_calc_straight(ase_atoms,charge=charge,uhf=uhf,method=method,solvent=solvent) 
    else:
        mol = io_molecule.convert_io_molecule(structure,detect_charge_spin=detect_charge_spin)
        ase_atoms = mol.ase_atoms
        if detect_charge_spin:
            set_XTB_calc_straight(ase_atoms,charge=mol.charge,uhf=mol.xtb_uhf,method=method,solvent=solvent) 
        else:
            set_XTB_calc_straight(ase_atoms,charge=charge,uhf=uhf,method=method,solvent=solvent) 
    good = True
    with arch_context_manage.make_temp_directory() as _:
        try:
            dyn = BFGSLineSearch(ase_atoms)
            dyn.run(fmax=fmax)
        except:
            good = False
    return ase_atoms,good


def xtb_sp(structure, charge=None, uhf=None, method='GFN2-xTB',solvent='none',
            detect_charge_spin=False):
    """xtb_sp singlepoint on the structure with xtb

    Parameters
    ----------
    structure : any 3D structure
       xyz, mol2string, ase atoms ...
    charge :int, optional
        total charge on the system, by default None
    uhf : int, optional
        number of unpaired electrons in the system, by default None
    method : str, optional
        which method to use, by default 'GFN2-xTB'
    solvent : str, optional
        any name xtb solvent, by default 'none'

    Returns
    -------
    ase_atoms : ase.atoms.Atoms
        structure with calculator/energy calculated
    good : bool
        whether the relaxation was succesful!
    """
    if isinstance(structure,ase.atoms.Atoms):
        ase_atoms = structure
        if ase_atoms.calc is None:
            set_XTB_calc_straight(ase_atoms,charge=charge,uhf=uhf,method=method,solvent=solvent) 
    else:
        mol = io_molecule.convert_io_molecule(structure,detect_charge_spin=detect_charge_spin)
        ase_atoms = mol.ase_atoms
        if detect_charge_spin:
            set_XTB_calc_straight(ase_atoms,charge=mol.charge,uhf=mol.xtb_uhf,method=method,solvent=solvent) 
        else:
            set_XTB_calc_straight(ase_atoms,charge=charge,uhf=uhf,method=method,solvent=solvent) 
    good = True
    with arch_context_manage.make_temp_directory() as _:
        try:
            ase_atoms.get_total_energy()
        except:
            good = False
    return ase_atoms,good


eV2Hartree = 1 / 27.2114

def get_rxyz_string(ase_atoms):
    """dump the ase_atoms to rxyz file string"""
    natom = len(ase_atoms)
    ss = ''
    # write an xyz file first
    ss += "%d\n\n" % natom
    for symb, coord in zip(ase_atoms.get_chemical_symbols(), ase_atoms.positions):
        ss += "%2s %12.6f %12.6f %12.6f\n" % (symb, *coord[:3])
    ss += "\n"
    # then force
    ss += "FORCES\n"
    for symb, force in zip(ase_atoms.get_chemical_symbols(), ase_atoms.get_forces()):
        ss += "%3s %22.14e %22.14e %22.14e\n" % (symb, *(force*eV2Hartree))
    ss += '\n'
    # then pbc, if needed
    # Potentially usefull - ase_atoms.cell, ase_atoms.get_pbc(), ase_atoms.get_cell_lengths_and_angles(), get_celldisp(), get_cell()
    # if ase_atoms.cell is not None:
    #     ss += "PBC\n"
    #     ss += "%12.6f %12.6f %12.6f\n" % (chemical.pbc.boxhi[0] -
    #         chemical.pbc.boxlo[0], 0.0, 0.0))
    #     ss += "%12.6f %12.6f %12.6f\n" % (chemical.pbc.xy,
    #         chemical.pbc.boxhi[1] - chemical.pbc.boxlo[1], 0.0))
    #     fp.write("%12.6f %12.6f %12.6f\n" % (chemical.pbc.xz, 
    #                                             chemical.pbc.yz, 
    #                     chemical.pbc.boxhi[2] - chemical.pbc.boxlo[2]))
    #     fp.write('\n')
    # then charge and spin
    ss += "ATOM-CHARGE-SPIN\n"
    for symb, charge, spin in zip(ase_atoms.get_chemical_symbols(), ase_atoms.get_charges(),
                                    ase_atoms.get_initial_magnetic_moments()):
        ss += "%2s %12.6f %12.6f\n" % (symb, charge, spin)
    ss += '\n'
    # then different properties
    ss += "ENERGY %22.14f\n" % (ase_atoms.get_total_energy()*eV2Hartree)
    ss += "PA_BINDING_ENERGY %22.14f\n" % (ase_atoms.get_total_energy()*eV2Hartree
                                            / len(ase_atoms))
    dipole_vect = ase_atoms.get_dipole_moment()
    ss += 'DIPOLE_VEC %12.6f %12.6f %12.6f\n' % (dipole_vect[0],
                                        dipole_vect[1],
                                        dipole_vect[2])
    ss += 'DIPOLE %12.6f\n'%np.linalg.norm(dipole_vect)
    ss += "CHARGE %d\n" % ase_atoms.get_initial_charges().sum()
    ss += "MULTIPLICITY %d\n" % round(sum(ase_atoms.get_initial_magnetic_moments)+1)
    # fp.write("DIFFAB %f\n" % sum(result['atomspin']))
    # fp.write('BAND_GAP %12.6f\n' % (result["egap"] * eV2Hartree))
    # fp.write('CONVERGED %s\n' % str(result["qconverged"]))
    # for prop in more_props:
    #     fp.write('%s %f\n' % (prop.upper(), result[prop]))
    ss += 'SOURCE %s\n' % ase_atoms.get_calculator().name
    return ss


def calc_xtb_ref_dict():
    energydict = dict()
    print('--------1---------')
    for i,elem in enumerate(io_ptable.elements):
        if i>0 and i<87:
            atoms = Atoms([Atom(elem,(0,0,0))])
            if elem in io_ptable.all_metals:
                if elem in io_ptable.lanthanides:
                    spin = 0
                    charge = 3
                else:
                    spin = io_ptable.metal_spin_dict[elem]
                    charge = io_ptable.metal_charge_dict[elem]
            else:
                if i % 2 == 0:
                    spin = 0
                else:
                    spin = 1 
                charge = 0
            set_XTB_calc_straight(atoms,charge=charge,uhf=spin)
            try:
                energy = atoms.get_total_energy()
                energydict[elem] = energy
            except:
                energydict[elem] = None
    print('--------2---------')
    for elem,spin in io_ptable.second_choice_metal_spin_dict.items():
        print(elem)
        if io_ptable.elements.index(elem) < 87:
            atoms = Atoms([Atom(elem,(0,0,0))])
            charge = io_ptable.metal_charge_dict[elem]
            set_XTB_calc_straight(atoms,charge=charge,uhf=spin)
            try: 
                energy = atoms.get_total_energy()
            except:
                energy = None  
            failed=False
            if (energydict[elem] is None) and isinstance(energy,float):
                energydict[elem] = energy
                print('Fixed {} with alternate spin'.format(elem))
            elif (energydict[elem] is None) and (energy is None):
                failed = True
                print('{}'.format(elem) + ' STILL FAILED')
            if (not failed) and (energy is not None):
                if energydict[elem] > energy:
                    energydict[elem] = energy
                    print('replaced {} with alternate spin'.format(elem))
    print('--------3---------')
    for elem,spin in io_ptable.metal_spin_dict.items():
        print(elem)
        if io_ptable.elements.index(elem) < 87:
            atoms = Atoms([Atom(elem,(0,0,0))])
            charge = 0
            if (i % 2 == 0) or (elem not in io_ptable.lanthanides):
                spin =0
            else:
                spin =1
            set_XTB_calc_straight(atoms,charge=charge,uhf=spin)
            try: 
                energy = atoms.get_total_energy()
            except:
                energy = None  
            failed=False
            if (energydict[elem] is None) and isinstance(energy,float):
                energydict[elem] = energy
                print('Fixed {} with 0 charge low spin'.format(elem))
            elif (energydict[elem] is None) and (energy is None):
                failed = True
                print('{}'.format(elem) + ' STILL FAILED')
            if (not failed) and (energy is not None):
                if energydict[elem] > (energy):
                    energydict[elem] = energy
                    print('replaced {} with 0 charge alternate spin'.format(elem))
    print('--------4---------')
    for elem,spin in io_ptable.metal_spin_dict.items():
        print(elem)
        if io_ptable.elements.index(elem) < 87:
            atoms = Atoms([Atom(elem,(0,0,0))])
            charge = 0
            if (i % 2 == 0) or (elem not in io_ptable.lanthanides):
                spin =2
            else:
                spin =3
            set_XTB_calc_straight(atoms,charge=charge,uhf=spin)
            try: 
                energy = atoms.get_total_energy()
            except:
                energy = None  
            failed=False
            if (energydict[elem] is None) and isinstance(energy,float):
                energydict[elem] = energy
                print('Fixed {} with 0 charge 2 spin'.format(elem))
            elif (energydict[elem] is None) and (energy is None):
                failed = True
                print('{}'.format(elem) + ' STILL FAILED')
            if (not failed) and (energy is not None):
                if energydict[elem] > energy:
                    energydict[elem] = energy
                    print('replaced {} with 0 charge 2 spin'.format(elem))
    print('--------5---------')
    for elem,spin in io_ptable.metal_spin_dict.items():
        print(elem)
        if io_ptable.elements.index(elem) < 87:
            atoms = Atoms([Atom(elem,(0,0,0))])
            charge = 0
            if (i % 2 == 0) or (elem not in io_ptable.lanthanides):
                spin=4
            else:
                spin=5
            set_XTB_calc_straight(atoms,charge=charge,uhf=spin)
            try: 
                energy = atoms.get_total_energy()
            except:
                print('Failed 5 spin',elem)
                energy = None  
            failed=False
            if (energydict[elem] is None) and isinstance(energy,float):
                energydict[elem] = energy
                print('Fixed {} with 0 charge 4 spin'.format(elem))
            elif (energydict[elem] is None) and (energy is None):
                failed = True
                print('{}'.format(elem) + ' STILL FAILED')
            if (not failed) and (energy is not None):
                if energydict[elem] > energy:
                    energydict[elem] = energy
                    print('replaced {} with 0 charge 5 spin'.format(elem))

    # Fix Mo
    atoms = Atoms([Atom("Mo",(0,0,0))])
    charge=0
    set_XTB_calc_straight(atoms,charge=charge,uhf=6)
    e = atoms.get_total_energy()
    energydict['Mo'] = e
    # energydict_scaled_ha = {sym:val/Hartree for sym,val in energydict.items()}
    return energydict