""" 
Code containing molecule/io
Molecule class encodes graph/bond order information and can handle a variety of types of inputs.
convert_io_molecule handles conversion to molecule object.

Also contains sanity check and graph sanity check routines for checking final geometries.

Developed by Michael Taylor
"""

import ase
from ase.io import (read, Trajectory)
from ase.atoms import Atoms
from ase.atom import Atom
import numpy as np
import re
import copy
import itertools
import architector
from architector import io_obabel
from architector.io_core import (Geometries, calc_all_coord_atom_angles)
import architector.io_ptable as io_ptable
from scipy.spatial.transform import Rotation as Rot
from io import StringIO
import pandas as pd
from scipy.sparse.csgraph import (csgraph_from_dense, connected_components)

def convert_io_molecule(structure,
                        detect_charge_spin=False,
                        charge=None,
                        uhf=None,
                        xtb_uhf=None,
                        xtb_charge=None):
    """convert_io_molecule
    Handle multiple types of structures passed and convert them to architector.io_molecule.Molecule objects.

    Parameters
    ----------
    structure : str
        Structure to convert to mol object
    detect_charge_spin : bool, optional
        Use openbabel and structure information to estimate a charge/spin state for the molecule
    charge : int, optional 
        charge on the molecule, default None
    uhf : int, optional
        unpaired electrons in the molecule, default None
    xtb_uhf : int, optional
        unpaired electrons in molecule for use by XTB, default None
    xtb_charge : int, optional
        charge used for XTB relaxation, default None
    """
    mol = Molecule()
    if isinstance(structure,(str,ase.atoms.Atoms)):
        # Mol2string
        if 'TRIPOS' in structure:
            # mol = structure
            mol.read_mol2(structure,readstring=True)
        # Xyz filename
        elif structure[-4:] == '.xyz':
            mol.read_xyz(structure,readstring=False)
        # rxyz filename
        elif structure[-5:] == '.rxyz':
            mol.read_rxyz(structure)
        # mol2 filename
        elif structure[-5:] == '.mol2':
            mol.read_mol2(structure,readstring=False)
        elif structure[-4:] == '.cif':
            obmol = io_obabel.convert_cif_obmol(structure,readstring=False)
            mol2 = io_obabel.convert_obmol_mol2(obmol)
            mol.read_mol2(mol2,readstring=True)
        elif structure[-5:] == '.traj': # Read in trajectory file.
            traj = Trajectory(structure)
            output = []
            for ats in traj:
                output.append(convert_io_molecule(ats))
            return output
        elif isinstance(structure,str) and (len(structure.split('\n')) > 3) and (structure.split('\n')[0].replace(' ','').isnumeric()) \
            and (('FORCES' in structure) or ('ENERGY' in structure)): # RXYZ string.
            mol.read_rxyz(structure,readstring=True)
        # checking for number at start of string -> indicates xyz string
        elif isinstance(structure,str) and (len(structure.split('\n')) > 3) and (structure.split('\n')[0].replace(' ','').isnumeric()):
            mol.read_xyz(structure,readstring=True)
        # checking for similar file without header
        elif isinstance(structure,str) and (len(structure.split('\n')[0].split()) == 4) and structure.split('\n')[0].split()[0]:
            mol.read_xyz(structure,readstring=True)
        elif isinstance(structure,str): # Smiles?
            try:
                tmol = io_obabel.get_obmol_smiles(structure)
                mol2 = io_obabel.convert_obmol_mol2(tmol)
                mol.read_mol2(mol2,readstring=True)
                mol.charge = tmol.GetTotalCharge()
                charges = np.array(np.zeros(len(mol.ase_atoms)))
                charges[0] = mol.charge
                mol.ase_atoms.set_initial_charges(charges)
            except:
                raise ValueError('Not Recognized Structure Type (str)!')
        elif isinstance(structure, ase.atoms.Atoms):
            mol.load_ase(structure,atom_types=structure.get_chemical_symbols())
        else:
            raise ValueError('Not Recognized Structure Type!')
        if detect_charge_spin:
            mol.detect_charge_spin()
            return mol
        else:
            if charge is not None:
                mol.charge = charge
                charges = np.zeros(len(mol.ase_atoms))
                charges[0] = mol.charge
                mol.ase_atoms.set_initial_charges(charges)
            if uhf is not None:
                mol.uhf = uhf
                uhf_vect = np.zeros(len(mol.ase_atoms))
                mol.ase_atoms.set_initial_magnetic_moments(uhf_vect)
            if xtb_uhf is not None:
                mol.xtb_uhf = xtb_uhf
            if xtb_charge is not None:
                mol.xtb_charge = xtb_charge
            return mol
    elif isinstance(structure,architector.io_molecule.Molecule):
        # struct = mol
        # for key,val in structure.__dict__.items():
        #         setattr(struct,key,val)
        structure.ase_atoms.calc = None
        struct = copy.deepcopy(structure)
        if detect_charge_spin:
            struct.detect_charge_spin()
            return struct
        else:
            if charge is not None:
                struct.charge = charge
                charges = np.zeros(len(struct.ase_atoms))
                charges[0] = struct.charge
                struct.ase_atoms.set_initial_charges(charges)
            if uhf is not None:
                struct.uhf = uhf
                uhf_vect = np.zeros(len(struct.ase_atoms))
                struct.ase_atoms.set_initial_magnetic_moments(uhf_vect)
            if xtb_uhf is not None:
                mol.xtb_uhf = xtb_uhf
            return struct
    else:
        raise ValueError('String needed for classifying structure type.')

def convert_ase_xyz(ase_atoms):
    """convert_ase_xyz

    Parameters
    ----------
    ase_atoms : ase.Atoms
        ase atoms to write to xyz string

    Returns
    -------
    outstring : str
        xyz file string
    """
    outstring = '{}\n\n'.format(len(ase_atoms))
    for atom in ase_atoms:
        outstring += '{} {} {} {}\n'.format(atom.symbol,
                                            atom.position[0],
                                            atom.position[1],
                                            atom.position[2])
    outstring = outstring.strip('\n')
    return outstring


def convert_xyz_ase(structure_str):
    """convert_xyz_ase

    Parameters
    ----------
    structure_str : str
        xyz file string

    Returns
    -------
    ase_atoms : ase.Atoms
        ase atoms to write to xyz string
    """
    return ase.io.read(StringIO(structure_str), format="xyz")


class Molecule:
    """Molecule

    Class containing all the relevant information about the molecules.
    Key attributes:

    self.ase_atoms : full ASE atoms object including positions, energy, forces, symbols...
    self.graph : molecular graph as Num_AtomsxNum_atoms numpy array of 0/1.
    self.BO_graph : dictionary representation of molecular graph with bond orders
    self.uhf : unpaired electrons (spin-multiplicity!) 
    self.charge : molecular charge 
    """

    def __init__(self, in_ase=False, BO_dict={}, atom_types=[], cell=[],
                 charge=None, uhf=None, xtb_uhf=None, xtb_charge=None,
                 actinides=None):
        """Molecule class

        Parameters
        ----------
        in_ase : bool, optional
            Ase atoms added, by default False
        BO_dict : dict, optional
            Bond orders dictionary, by default {}
        atom_types : list, optional
            Atom types - SYBYL, by default []
        cell : list, optional
            Unit cell, by default []
        charge : int, optional
            molecular charge, by default None
        uhf : int, optional
            number unpaired electrons, by default None
        xtb_uhf : int, optional
            number of unpaired electrons for XTB, by default None
        xtb_charge : int, optional
            charge on molecule for XTB, by default None
        actinides : None | list, optional
            indices of the actinides, by default None
        """
        self.dists_sane = True
        self.sanity_check_dict = {}
        self.ase_constraints = {} ### Add distance constraints here.
        self.actinides_swapped = False
        if isinstance(in_ase,ase.atoms.Atoms):
            self.ase_atoms = in_ase.copy()
            self.BO_dict = BO_dict
            self.charge = charge
            self.uhf = uhf
            self.xtb_uhf = xtb_uhf
            self.xtb_charge = xtb_charge
            self.atom_types = atom_types
            self.actinides = actinides
            self.cell = cell
            if len(BO_dict) > 0:
                self.graph = np.zeros((len(self.ase_atoms),len(self.ase_atoms)))
                for key,_ in self.BO_dict.items():
                    i = int(key[0]) - 1 # BO Dict is 1-index (thanks to  OBmol/mol2 format)
                    j = int(key[1]) - 1
                    self.graph[i,j] = 1 
                    self.graph[j,i] = 1 
            else:
                self.graph = []
        elif isinstance(in_ase,str) and (in_ase[-4:] == '.xyz'):
            self.read_xyz(in_ase)
            self.BO_dict = BO_dict
            self.atom_types = atom_types
            self.charge = charge
            self.uhf = uhf
            self.xtb_uhf = xtb_uhf
            self.xtb_charge = xtb_charge
            self.cell = cell
            if len(BO_dict) > 0:
                self.graph = np.zeros((len(self.ase_atoms),len(self.ase_atoms)))
                for key,_ in self.BO_dict.items():
                    i = int(key[0]) - 1 # BO Dict is 1-index (thanks to  OBmol/mol2 format)
                    j = int(key[1]) - 1
                    self.graph[i,j] = 1 
                    self.graph[j,i] = 1 
            else:
                self.graph = []
        elif isinstance(in_ase,str) and (in_ase[-5:] == '.mol2'):
            self.read_mol2(in_ase)
            self.actinides = [i for i,x in enumerate(self.ase_atoms.get_chemical_symbols()) if x in io_ptable.actinides]
        elif isinstance(in_ase,bool):
            self.ase_atoms = None
            self.actinides = None
        else:
            raise ValueError('Need ase.atoms.Atoms/xyz/mol2 as input for molecule class!')

    def load_ase(self,in_ase, BO_dict=dict(), atom_types=[], cell=[],
                charge=None, uhf=None, xtb_uhf=None, xtb_charge=None):
        """load_ase read in ase atoms object.

        Parameters
        ----------
        in_ase : ase.atoms.Atoms
            structure
        BO_dict : dict, optional
            bond order dictionary, by default dict()
        atom_types : list, optional
            list of atom types (ideally sybyl style), by default []
        cell : list, optional
            unit cell, by default []
        charge : int/float, optional
            charge on the system, by default None
        uhf : int/float, optional
            number unpaired electrons on the system, by default None
        xtb_uhf : int/float, optional
            number unpaired electrons on the system desired for XTB, by default None

        Raises
        ------
        ValueError
            If ase atoms not passed.
        """
        if isinstance(in_ase,ase.atoms.Atoms):
            self.ase_atoms = in_ase.copy()
        else:
            raise ValueError('Need ase.atoms.Atoms as input for molecule class!')
        self.BO_dict = BO_dict.copy()
        self.atom_types = atom_types
        self.actinides = [i for i,x in enumerate(self.ase_atoms.get_chemical_symbols()) if x in io_ptable.actinides]
        self.cell = cell
        if charge is not None:
            self.charge = charge
        else:
            self.charge = np.sum(self.ase_atoms.get_initial_charges())
        if uhf is not None:
            self.uhf = uhf
        else:
            self.uhf = np.sum(self.ase_atoms.get_initial_magnetic_moments())
        if xtb_uhf is not None:
            self.xtb_uhf = xtb_uhf
        else:
            self.xtb_uhf = int(np.sum(self.ase_atoms.get_initial_magnetic_moments()))
        if xtb_charge is not None:
            self.xtb_charge = xtb_charge
        else: 
            self.xtb_charge = np.sum(self.ase_atoms.get_initial_charges())
        if len(self.BO_dict) > 0:
            self.graph = np.zeros((len(self.ase_atoms),len(self.ase_atoms)))
            for key,_ in self.BO_dict.items():
                i = int(key[0]) - 1 # BO Dict is 1-index (thanks to  OBmol/mol2 format)
                j = int(key[1]) - 1
                self.graph[i,j] = 1 
                self.graph[j,i] = 1 
        else:
            self.graph = []
            
    def write_xyz(self, filename, writestring=False):
        """convert_ase_xyz

        Parameters
        ----------
        ase_atoms : ase.Atoms
            ase atoms to write to xyz string

        Returns
        -------
        outstring : str
            xyz file string
        """
        ase_atoms = self.ase_atoms
        outstring = '{}\n\n'.format(len(ase_atoms))
        for atom in ase_atoms:
            outstring += '{} {} {} {}\n'.format(atom.symbol,
                                                atom.position[0],
                                                atom.position[1],
                                                atom.position[2])
        outstring = outstring.strip('\n')
        if writestring:
            return outstring
        else:
            if filename[-4:] == '.xyz':
                filename = filename
            else:
                filename = filename.replace('.','') + '.xyz'
            with open(filename,'w') as file1:
                file1.write(outstring)

    def read_xyz(self,filename,readstring=False):
        """read_xyz read in an xyz file

        Parameters
        ----------
        filename : str
            name of file to read from
        readstring : bool, optional
            whether to read from a string, by default False
        """
        if readstring:
            filename = StringIO(filename)
            self.ase_atoms = read(filename,format='xyz',parallel=False)
        else:
            self.ase_atoms = read(filename,parallel=False)
        self.actinides = [i for i,x in enumerate(self.ase_atoms.get_chemical_symbols()) if x in io_ptable.actinides]
        self.graph = []
        self.BO_dict = {}
        self.charge = None
        self.uhf = None
        self.xtb_charge = None
        self.xtb_uhf = None
        self.atom_types = [x.symbol for x in self.ase_atoms]
        self.cell = []

    def read_rxyz(self,filename,readstring=False):
        """read_rxyz read an RXYZ file
        - Specific to LANL

        Parameters
        ----------
        filename : str
            name of file to read from
        readstring : bool, optional
            whether to read from a string, by default False
        """
        if not readstring:
            with open(filename,'r') as file1:
                lines = file1.readlines()
        else:
            lines = filename.split('\n')
        atoms = []
        for line in lines:
            sline = line.split()
            if 'FORCES' in line:
                break
            elif len(sline) == 4:
                symbol = sline[0]
                coords = (float(sline[1]),float(sline[2]),float(sline[3]))
                if len(atoms) == 0:
                    atoms = Atoms([Atom(symbol,coords)])
                else:
                    atoms.append(Atom(symbol,coords))
        self.ase_atoms = atoms
        self.actinides = [i for i,x in enumerate(self.ase_atoms.get_chemical_symbols()) if x in io_ptable.actinides]
        self.graph = []
        self.charge = None
        self.uhf = None
        self.xtb_charge = None
        self.xtb_uhf = None
        self.BO_dict = {}
        self.atom_types = [x.symbol for x in self.ase_atoms]
        self.cell = []

    def write_mol2(self, filename, writestring=False):
        """write_mol2 routine for writing a mol2 file.
        Works with integrating with CSD software and openbabel.

        Parameters
        ----------
        filename : str
            name of file to read from
        writestring : bool, optional
            whether to write to string, by default False

        Returns
        -------
        ss : str
            mol2 file in string format
        """
        if len(self.graph) < 1:
            self.create_mol_graph()
        natoms = len(self.ase_atoms)
        csg = csgraph_from_dense(self.graph)
        disjoint_components = connected_components(csg)
        if disjoint_components[0] > 1:
            atom_group_names = ['RES'+str(x+1) for x in disjoint_components[1]]
            atom_groups = [str(x+1) for x in disjoint_components[1]]
        else:
            atom_group_names = ['RES1']*natoms
            atom_groups = [str(1)]*natoms
        charges = np.zeros(natoms)
        charge_string = 'NoCharges'
        if (self.charge is not None) and (self.uhf is not None) and (self.xtb_charge is not None) and (self.xtb_uhf is not None):
            ss = '@<TRIPOS>MOLECULE\n{} Charge: {} Unpaired_Electrons: {} XTB_Unpaired_Electrons: {} XTB_Charge: {}\n'.format(filename,
                    int(self.charge),int(self.uhf),int(self.xtb_uhf),int(self.xtb_charge))
        else:
            ss = '@<TRIPOS>MOLECULE\n{}\n'.format(filename)            
        ss += ' {0:5d} {1:5d} {2:5d} {3:5d} {4:5d}\n'.format(natoms,
                                    int(csg.nnz/2), disjoint_components[0],
                                    0, 0)
        ss += 'SMALL\n'
        ss += charge_string + '\n' + '****\n' + 'Generated from Architector\n\n'
        ss += '@<TRIPOS>ATOM\n'
        atom_default_dict = {'C': '3', 'N': '3', 'O': '2', 'S': '3', 'P': '3'}
        atom_types = self.ase_atoms.get_chemical_symbols()
        atom_type_numbers = np.ones(len(atom_types))
        atom_types_mol2 = []
        for i, atom in enumerate(self.ase_atoms):
            if atom.symbol != self.atom_types[i]:
                atom_types_mol2 = self.atom_types[i]
            elif atom.symbol in list(atom_default_dict.keys()):
                atom_types_mol2 = atom.symbol + '.' + atom_default_dict[atom.symbol]
            else:
                atom_types_mol2 = atom.symbol
            type_ind = atom_types.index(atom.symbol)
            atom_coords = atom.position
            ss += '{0:6d} {1:6s} {2:9.4f} {3:9.4f} {4:9.4f}   {5:6s}{6:5d} {7:5s}{8:8.4f}\n'.format(
                i+1, atom.symbol+str(int(atom_type_numbers[type_ind])), 
                atom_coords[0], atom_coords[1], atom_coords[2], 
                atom_types_mol2, int(atom_groups[i]),
                atom_group_names[i], charges[i]
            )
            atom_type_numbers[type_ind] += 1
        ss += '@<TRIPOS>BOND\n'
        bonds = csg.nonzero()
        bond_count = 1
        if self.BO_dict:
            bondorders = True
        else:
            bondorders = False
        for i, b1 in enumerate(bonds[0]):
            b2 = bonds[1][i]
            if b2 > b1 and not bondorders:
                ss += '{0:6d}{1:6d}{2:6d}{3:>5s}\n'.format(
                    bond_count, b1+1, b2+1, str(1)
                )
                bond_count += 1
            elif b2 > b1 and bondorders:
                ss += '{0:6d}{1:6d}{2:6d}{3:>5s}\n'.format(
                    bond_count, b1+1, b2+1, str(self.BO_dict[(int(b1)+1, int(b2)+1)])
                )
                bond_count += 1
        ss += '@<TRIPOS>SUBSTRUCTURE\n'
        unique_group_names = np.unique(atom_group_names)
        for i, name in enumerate(unique_group_names):
            ss += '{0:6d} {1:6s}{2:7d} GROUP             0 ****  ****    0  \n'.format(
                i+1, name, atom_group_names.count(name) 
            )

        if hasattr(self,'cell'):
            # If there's a cell, write it - else 'ehh'
            if (len(self.cell) == 6): #Only abc,alphabetagamma defined (add 1 1 for space group)
                ss += '@<TRIPOS>CRYSIN\n'
                ss += '{0:10.4f}{1:10.4f}{2:10.4f}{3:10.4f}{4:10.4f}{5:10.4f}{6:6d}{7:6d}\n'.format(
                    self.cell[0], self.cell[1], self.cell[2],
                    self.cell[3], self.cell[4], self.cell[5],
                    '1', '1'
                    )
            elif (len(self.cell) == 8): #abc,alphabetagamma and space group defined
                ss += '@<TRIPOS>CRYSIN\n'
                ss += '{0:10.4f}{1:10.4f}{2:10.4f}{3:10.4f}{4:10.4f}{5:10.4f}{6:6d}{7:6d}\n'.format(
                    self.cell[0], self.cell[1], self.cell[2],
                    self.cell[3], self.cell[4], self.cell[5],
                    self.cell[6], self.cell[7]
                    )

        if writestring:
            return ss
        else:
            if '.mol2' not in filename:
                if '.' not in filename:
                    filename += '.mol2'
                else:
                    filename = filename.split('.')[0]+'.mol2'
            with open(filename, 'w') as file1:
                file1.write(ss)

    def read_mol2(self, filename, readstring=False):
        """ Read mol2 into a Mol class instance. Stores the bond orders and atom types (SYBYL).
        Will read in charge and spin states assigned to complexes generated from architector to the molecule!

        Parameters
        -------
        filename : string
            String of path to XYZ file. Path may be local or global. May be read in as a string.
        readstring : bool, optional
            Flag for deciding whether a string of mol2 file is being passed as the filename, default False
        """
        # ptable=PTable()
        graph = False
        bo_dict = False
        self.atom_types = []
        if readstring:
            s = filename.splitlines()
        else:
            with open(filename, 'r') as f:
                s = f.read().splitlines()
        read_atoms = False
        read_bonds = False
        read_cell = False
        charge = None
        spin = None
        xtb_spin = None
        xtb_charge = None
        for line in s:
            # Get Atoms First
            if ('Charge:' in line) and ('Unpaired_Electrons:' in line):
                charge = float(line.split()[2])
                spin = int(line.split()[4])
                xtb_spin = int(line.split()[6])
                xtb_charge = int(line.split()[8])
            if ('<TRIPOS>BOND' in line) or ('<TRIPOS>UNITY_ATOM_ATTR' in line):
                read_atoms = False
            if ('<TRIPOS>SUBSTRUCTURE' in line) or ('<TRIPOS>UNITY_ATOM_ATTR' in line):
                read_bonds = False
                read_atoms = False
            if read_atoms:
                s_line = line.split()
                # Check redundancy in Chemical Symbols
                atom_symbol1 = re.sub('[0-9]+[A-Z]+', '', line.split()[1])
                atom_symbol1 = re.sub('[0-9]+', '', atom_symbol1)
                atom_symbol2 = line.split()[5]
                if len(atom_symbol2.split('.')) > 1:
                    atype = atom_symbol2
                else:
                    atype = False
                atom_symbol2 = atom_symbol2.split('.')[0]
                if atom_symbol1 in io_ptable.elements:
                    atom = ase.atom.Atom(atom_symbol1, [float(s_line[2]), float(
                        s_line[3]), float(s_line[4])])
                    if isinstance(atype,str):
                        self.atom_types.append(atype)
                    else:
                        self.atom_types.append(atom_symbol1)
                elif atom_symbol2 in io_ptable.elements:
                    atom = ase.atom.Atom(atom_symbol2, [float(s_line[2]), float(
                        s_line[3]), float(s_line[4])])
                    if isinstance(atype,str):
                        self.atom_types.append(atype)
                    else:
                        self.atom_types.append(atom_symbol2)
                else:
                    raise ValueError('Cannot find atom symbol in ptable')
                if isinstance(self.ase_atoms,ase.atoms.Atoms):
                    self.ase_atoms.append(atom)
                else:
                    self.ase_atoms = ase.atoms.Atoms([atom])
            if read_bonds:  # Read in bonds to molecular graph
                s_line = line.split()
                graph[int(s_line[1]) - 1, int(s_line[2]) - 1] = 1
                graph[int(s_line[2]) - 1, int(s_line[1]) - 1] = 1
                bo_dict[tuple(
                    sorted([int(s_line[1]), int(s_line[2])]))] = s_line[3]
            if read_cell:
                s_line = line.split()
                self.cell = [float(s_line[0]), float(s_line[1]), float(s_line[2]),
                        float(s_line[3]), float(s_line[4]), float(s_line[5]),
                        int(s_line[6]),int(s_line[7])]
            if '<TRIPOS>ATOM' in line:
                read_atoms = True
            if '<TRIPOS>BOND' in line:
                read_bonds = True
                # initialize molecular graph
                graph = np.zeros((len(self.ase_atoms), len(self.ase_atoms)))
                bo_dict = dict()
            if '<TRIPOS>CRYSIN' in line:
                read_cell = True
        if isinstance(graph, np.ndarray):  # Enforce mol2 molecular graph if it exists
            self.graph = graph
            self.BO_dict = bo_dict
        else:
            self.graph = []
            self.BO_dict = {}
        self.actinides = [i for i,x in enumerate(self.ase_atoms.get_chemical_symbols()) if x in io_ptable.actinides]
        self.charge = charge
        self.uhf = spin
        self.xtb_uhf = xtb_spin
        self.xtb_charge = xtb_charge

    def detect_charge_spin(self):
        """detect_charge_spin 
        Detect the spin/charge of the complex using openbabel intepretations of bond orders
        and partial charges. Definitely do not guarantee this is correct for most cases.
        """
        metals = np.array(self.find_metals())
        # Metal charges
        if len(metals) > 0:
            charge = np.sum([io_ptable.metal_charge_dict[x] for x in np.array(self.ase_atoms.get_chemical_symbols())[metals]])
            mol2str = self.write_mol2('thing.mol2',writestring=True)
            _, _, infodict = io_obabel.obmol_lig_split(mol2str,return_info=True,calc_coord_atoms=False)
            charge += np.sum(infodict['lig_charges'])
            met_syms = np.array(self.ase_atoms.get_chemical_symbols())[metals]
            uhf = np.sum([io_ptable.metal_spin_dict[x] for x in met_syms])
        else: # Assume low-spin and that charge was already assinged at some step.
            if self.charge is not None:
                charge = self.charge
            else:
                charge = np.sum(self.ase_atoms.get_initial_charges())
            uhf = 0
            met_syms = []
        even_odd_electrons = (np.sum([atom.number for atom in self.ase_atoms])-charge) % 2
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
        xtb_uhf = copy.deepcopy(uhf)
        if np.any([True for x in met_syms if x in io_ptable.heavy_metals]):
            xtb_uhf = 0
            if (even_odd_electrons == 1):
                xtb_uhf = 1
        self.charge = int(charge)
        self.uhf = int(uhf)
        self.xtb_uhf = int(xtb_uhf)
        self.xtb_charge = int(charge)

    def find_metal(self, debug=False):
        """find_metal
        pull out the metal index in a complex

        Returns
        -------
        metalind : int
            index of the metal or first metal if present.
        """
        syms = self.ase_atoms.get_chemical_symbols()
        metalinds = [i for i,x in enumerate(syms) if x in io_ptable.all_metals]
        if len(metalinds) == 1:
            metalind = metalinds[0]
        elif len(metalinds) > 1:
            if debug:
                print('Assigning first metal as metalind.')
            metalind = metalinds[0]
        else:
            if debug:
                print('No metals in this molecule.')
            metalind = None
        return metalind

    def find_metals(self):
        """find_metals find all metals in a molecule

        Returns
        -------
        metals : list
            indices of all the metals in the molecule
        """
        metals = [i for i, x in enumerate(self.ase_atoms) if (x.symbol in io_ptable.all_metals)]
        return metals

    def remove_atom(self, ind):
        """remove_atom delete an atom from the molecule with ind

        Parameters
        ----------
        ind : int
            index to remove
        """
        del self.ase_atoms[ind]
        if len(self.graph) > 0:
            self.graph = np.delete(np.delete(self.graph, ind, 0), ind, 1)
        del self.atom_types[ind]
        BO_dict_tmp = {x: y for x, y in self.BO_dict.items() if (ind+1 not in x)}
        bo_dict_out = dict()
        for x, y in BO_dict_tmp.items():
            newx = []
            if x[0] > ind+1:
                newx.append(x[0] - 1)
            else:
                newx.append(x[0])
            if x[1] > ind+1:
                newx.append(x[1] - 1)
            else:
                newx.append(x[1])
            newx = tuple(newx)
            bo_dict_out[newx] = y
        self.BO_dict = bo_dict_out

    def remove_metals(self):
        """remove_metals
        remove all metals from the molecule
        """
        metals = self.find_metals()
        for i in sorted(metals)[::-1]:
            self.remove_atom(i)

    def remove_hydrogens(self):
        """remove hydrogens from the model
        """
        hydrogens = [i for i,x in enumerate(self.ase_atoms.get_atomic_numbers()) if x == 1]
        for i in sorted(hydrogens)[::-1]:
            self.remove_atom(i)

    def remove_atoms(self, inds):
        """Delete the indices passed from the molecule.

        Parameters
        ----------
        inds : list/np.ndarray(int)
            Delete these indices from the molecule.
        """
        if isinstance(inds,(list,np.ndarray)):
            delete_inds = sorted(inds)[::-1] # Do these in reverse order.
            for ind in delete_inds:
                try:
                    self.remove_atom(int(ind))
                except:
                    print('Warning - type unknown for index {}. Need an integer got {}! This atom was not removed.'.format(
                        ind,type(ind)))
        else:
            print('Warning - either unknown list/array of indices to delete passed {}. No atoms were removed.'.format(
                inds))
                
    def create_mol_graph(self, cutoffs=True, skin=0.2, allow_mm_bonds=False):
        """create_mol_graph
        Create molecular graph based on cutoffs, default skin=0.2

        Parameters
        ----------
        cutoffs : bool, optional
            pass in a list of cutoffs for the structure?, by default True
        skin : float, optional
            tolerance for considering neighbors, by default 0.2
        allow_mm_bonds : bool, optional
            allow metal-metal bonds?, by default False
        """
        if cutoffs:
            cutoffsvect=np.array([io_ptable.rcov1[atom.number] for atom in self.ase_atoms])
        metals = self.find_metals()
        # Broadcast cutoffs sum
        cutoff_dist_mat = (cutoffsvect + skin) + (cutoffsvect+skin)[:,None]
        graph = np.zeros((len(self.ase_atoms), len(self.ase_atoms)))
        coords = self.ase_atoms.get_positions()
        # Calculate interatomic distances.
        act_dist_mat = np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
        delta_dist_mat = act_dist_mat - cutoff_dist_mat
        graph[np.where(delta_dist_mat < 0)] = 1
        graph = graph-np.eye(len(self.ase_atoms))
        if (len(metals) > 0):
            if (len(metals) > 1) and (not allow_mm_bonds):
                for i,j in itertools.combinations(metals,2):
                    graph[i,j] = 0  
                    graph[j,i] = 0

        # # Remove self interaction/bonding
        self.graph = graph
        self.create_BO_dict(cutoffs=cutoffs,skin=skin)

    def create_BO_dict(self, cutoffs=False, skin=0.2):
        """create_BO_dict
        Create BO dict - default everything to bond order 1.

        Parameters
        ----------
        cutoffs : bool, optional
            pass in a list of cutoffs for the structure?, by default True
        skin : float, optional
            tolerance for considering neighbors, by default 0.2
        """
        if len(self.graph) < 1:
            self.create_mol_graph(cutoffs=cutoffs, skin=skin)
        csg = csgraph_from_dense(self.graph)
        bonds = csg.nonzero()
        bond_dict = dict()
        for i, b1 in enumerate(bonds[0]):
            b2 = bonds[1][i]
            if b2 > b1:
                bond_dict[(b1+1,b2+1)] = '1'
        self.BO_dict = bond_dict

    def create_graph_from_bo_dict(self):
        """create_graph_from_bo_dict 
        Routine to create a molecular graph directly from a dictionary of bond orders
        """
        if len(self.BO_dict) > 0:
            self.graph = np.zeros((len(self.ase_atoms),len(self.ase_atoms)))
            for key,_ in self.BO_dict.items():
                i = int(key[0]) - 1 # BO Dict is 1-index (thanks to  OBmol/mol2 format)
                j = int(key[1]) - 1
                self.graph[i,j] = 1 
                self.graph[j,i] = 1 
        else:
            self.graph = []

    def append_ligand(self, ligand=None, non_coordinating=False):
        """append_ligand Add ligand to the structure
        Account for charges on the ligand.

        Parameters
        ----------
        ligand : dict, optional
            ligand to add to complex, by default None
        """
        lig_bo_dict = ligand['bo_dict']
        lig_ase_atoms = ligand['ase_atoms']
        lig_atom_types = ligand['atom_types']
        lig_constraints = ligand.get('ca_metal_dist_constraints',None)
        natoms = len(self.ase_atoms)
        newligbodict = dict()
        for key,val in lig_bo_dict.items():
            newkey = [0,0]
            if non_coordinating:
                newkey[0] = natoms + int(key[0])
            elif (int(key[0]) > 1):
                newkey[0] = natoms + int(key[0]) - 1
            else:
                newkey[0] = 1
            if non_coordinating:
                newkey[1] = natoms + int(key[1])
            elif (int(key[1]) > 1):
                newkey[1] = int(key[1]) + natoms - 1
            else:
                newkey[1] = 1
            newkey = tuple(newkey)
            newligbodict.update({newkey:val})
        if lig_constraints is not None:
            for ind,dist in lig_constraints.items():
                newind = natoms + ind
                self.ase_constraints.update({(0,newind):dist})
        self.BO_dict.update(newligbodict)
        self.ase_atoms += lig_ase_atoms
        self.atom_types += lig_atom_types
        if non_coordinating: # Assume all these are additive without covalent bonds!
            self.uhf += ligand['uhf']
            self.charge += ligand['charge']
            self.xtb_uhf += ligand['xtb_uhf']
            self.xtb_charge += ligand['xtb_charge']
        else:
            lcs = lig_ase_atoms.get_initial_charges().sum()
            self.charge += lcs
            self.xtb_charge += lcs
        self.create_graph_from_bo_dict()

    def graph_sanity_checks(self, factor=1.45, params={}, assembly=False):
        """graph_sanity_checks
        Check if any part of the molecule is blown up relative to the imposed molecular graph

        Parameters
        ----------
        factor : float, optional
            tolerance for long bonds - factor*(sum of covalent radii), by default 1.4
        params : dict, optional
            parameters from inputDict, default {}
        assembly : bool, optional
            if this is an assembly check or final check, default False -> final cutoffs.

        Returns
        -------
        sane : bool
            If the graph distances are longer than the factor -> indicating relatively garbage
            geometry.
        """
        run_check = True
        if (len(params) > 0) and (not assembly):
            run_check = params.get('full_sanity_checks',run_check)
            factor = params.get('full_graph_sanity_cutoff',factor)
        elif (len(params) > 0):
            run_check = params.get('assemble_sanity_checks',run_check)
            factor = params.get('assemble_graph_sanity_cutoff',factor)
        else:
            params = {'covrad_metal':None}
        if (len(self.graph) == 0) and (len(self.BO_dict) == 0):
            self.create_mol_graph()
            self.create_BO_dict()
        elif (len(self.BO_dict) > 0):
            self.create_graph_from_bo_dict()
        elif len(self.graph) > 0:
            self.create_BO_dict()
        sane = self.dists_sane
        graph_dists_dict = {}
        if run_check:
            if len(self.ase_atoms) > 1: # Don't test distances for single atom
                posits = self.ase_atoms.get_positions()
                m_inds = [i for i,x in enumerate(self.ase_atoms.get_chemical_symbols()) if x in io_ptable.all_metals]
                m_ind=None
                if len(m_inds) > 1:
                    if params.get('debug',False):
                        print('Warning - Sanity check with custom covrad only for mononuclear so far. Setting Defaults for all metals.')
                        multi=True
                elif len(m_inds) == 1:
                    m_ind = m_inds[0]
                    multi=False
                else:
                    multi=False
                mrad = None
                if isinstance(params['covrad_metal'],float):
                    mrad = params['covrad_metal']
                if np.any(np.isnan(posits)): # Any nan in positions
                    sane = False
                    graph_dists_dict.update({'containsNAN_posit':True})
                else:
                    all_dists = self.ase_atoms.get_all_distances()
                    cov_radii = np.array([io_ptable.rcov1[x] for x in self.ase_atoms.get_atomic_numbers()]) 
                    if (not (m_ind is None)) and (not (mrad is None)) and (not multi):
                        if mrad >= cov_radii[m_ind]:
                            cov_radii[m_ind] = mrad
                    for key, _ in self.BO_dict.items():
                        i = key[0] - 1
                        j = key[1] - 1
                        if all_dists[i,j] > factor*(cov_radii[i] + cov_radii[j]):
                            sane = False
                            graph_dists_dict.update({'Cutoff':factor})
                            graph_dists_dict.update({(i,j):all_dists[i,j]/(cov_radii[i] + cov_radii[j])})
                            if params.get('debug',False):
                                print('Graph distance long: ', all_dists[i,j]/(cov_radii[i] + cov_radii[j]))
                            break
        self.dists_sane = sane
        self.sanity_check_dict.update({'Graph_Dist_Checks':graph_dists_dict})

    def dist_sanity_checks(self,
                 smallest_dist_cutoff=0.55,
                 min_dist_cutoff=3,
                 params={},
                 pair_cutoffs=None,
                 assembly=False,
                 covrad_metal=None,
                 debug=False):
        """dist_sanity_checks
        Perform basic distance-based sanity checks on structure

        Parameters
        -------
        atoms : ase.Atoms
            atoms to check for sanity.
        params : dict, optional
            parameters for dictionary, default {}.
        pair_cutoffs : dict, optional
            pair distance cutoffs definition e.g. {('O','H'):0.8}, default None
        smallest_dist_cutoff : float
            distance cutoff-make sure sum of cov radii larger than dist*smallest_dist_cutoff
        min_dist_cutoff : int/float
            make sure all atoms are at least min_dist_cutoff from ANY other atom
        assembly : bool
            whether this is an assembly step or final relaxation
        covrad_metal : float
            the covalent radii of the metal if requested
        debug : bool
            print if debug requested.
        """
        run_check = True
        if (len(params) > 0) and (not assembly):
            run_check = params.get('full_sanity_checks', run_check)
            smallest_dist_cutoff = params.get('full_smallest_dist_cutoff',
                                              smallest_dist_cutoff)
            min_dist_cutoff = params.get('full_min_dist_cutoff',
                                         min_dist_cutoff)
        elif (len(params) > 0):
            run_check = params.get('assemble_sanity_checks', run_check)
            smallest_dist_cutoff = params.get('assemble_smallest_dist_cutoff',
                                              smallest_dist_cutoff)
            min_dist_cutoff = params.get('assemble_min_dist_cutoff',
                                         min_dist_cutoff)
        if debug:
            params.update({'debug': debug})
        if isinstance(pair_cutoffs, dict):
            new_pair_dict = dict()
            for key, val in pair_cutoffs.items():
                newkey = tuple(sorted(key))
                if newkey in new_pair_dict:
                    if params.get('debug', False):
                        print("Warning: multiple distance cutoffs passed for the same set of atom pairs")
                        print('Using the first one only!')
                # 'dist' or 'scale_covrad'
                elif key == 'type': 
                    new_pair_dict[key] = val
                else:
                    new_pair_dict[newkey] = val
            # Default to scale factors to the sum of covalent radii
            if 'type' not in new_pair_dict:
                new_pair_dict['type'] = 'scale_covrad'
        sane = self.dists_sane
        min_dist_dict = {}
        smallest_dist_dict = {}
        symbols = self.ase_atoms.get_chemical_symbols()
        if run_check:
            atoms = self.ase_atoms.copy()
            if len(atoms) > 1:  # Don't test distances for single atom
                posits = atoms.get_positions()
                m_inds = [i for i, x in enumerate(atoms.get_chemical_symbols()) if x in io_ptable.all_metals]
                m_ind = None
                if len(m_inds) > 1:
                    if params.get('debug',False):
                        print('Warning - Sanity check with custom covrad only for mononuclear so far. Setting Defaults for all metals.')
                        multi = True
                elif len(m_inds) == 1:
                    m_ind = m_inds[0]
                    multi = False
                else:
                    multi = False
                mrad = None
                if isinstance(covrad_metal, float):
                    mrad = covrad_metal
                elif isinstance(params, dict):
                    if isinstance(params.get('covrad_metal',False),float):
                        mrad = params['covrad_metal']
                if np.any(np.isnan(posits)): # Any nan in positions
                    if params.get('debug', False):
                        print('Nan in positions.')
                    sane = False
                else:
                    all_dists = atoms.get_all_distances()
                    cov_radii = np.array([io_ptable.rcov1[x] for x in atoms.get_atomic_numbers()])
                    if (not (m_ind is None)) and (not (mrad is None)) and (not multi):
                        cov_radii[m_ind] = mrad
                    for i in range(0, len(atoms)):
                        j_list = list(range(0, len(atoms)))
                        j_list.remove(i)
                        i_dists = []
                        for j in j_list:
                            i_dists.append(all_dists[i, j])
                            # Use pair_cutoffs for sanity check.
                            if isinstance(pair_cutoffs, dict):
                                if new_pair_dict['type'] == 'scale_covrad':
                                    cutoff = new_pair_dict.get(tuple(sorted(
                                        (symbols[i], symbols[j]))),
                                        smallest_dist_cutoff)*(cov_radii[i] + cov_radii[j])
                                elif new_pair_dict['type'] == 'dist':
                                    cutoff = new_pair_dict.get(tuple(sorted(
                                        (symbols[i], symbols[j]))),
                                        smallest_dist_cutoff*(cov_radii[i] + cov_radii[j]))
                                else:
                                    if params.get('debug', False):
                                        print('Warning: pair distance cutoffs for "{}" not implemented'.format(
                                            new_pair_dict.get('type', 'None')
                                        ))
                                        print('Acceptable types: "dist" or "scale_covrad!')
                            else:
                                cutoff = smallest_dist_cutoff*(cov_radii[i] + cov_radii[j])
                            if all_dists[i, j] < cutoff:  # Check for extra-crowded metals
                                sane = False
                                smallest_dist_dict.update({'Cutoff': cutoff})
                                smallest_dist_dict.update({(i, j): all_dists[i, j] / (
                                    cov_radii[i] + cov_radii[j])})
                                if params.get('debug', False):
                                    print('Dist short: ', all_dists[i, j])
                        if min(i_dists) > min_dist_cutoff:  # Catch cases where atom
                            # shot off metal center or blown up structure
                            sane = False
                            min_dist_dict.update({'Cutoff': min_dist_cutoff})
                            min_dist_dict.update({i: min(i_dists)})
                            if params.get('debug', False):
                                print('Mindist long: ', min(i_dists))
        self.dists_sane = sane
        self.sanity_check_dict.update({'Smallest_Dist_Checks': smallest_dist_dict,
                                       'Min_Dist_Checks': min_dist_dict})

    def get_can_label(self):
        """ Get molecular graph determinant - serves as unique identifier. 

        Returns
        -------
        safedet : str
            String containing the molecular graph determinant.
        """
        if not len(self.graph):
            self.create_mol_graph()
        syms = self.ase_atoms.get_chemical_symbols()
        weights = [io_ptable.masses[io_ptable.elements.index(x)] for x in syms]
        inds = np.nonzero(self.graph)
        tmpgraph = self.graph.copy()
        for j in range(len(syms)): # Set diagonal to weights
            tmpgraph[j, j] = weights[j]
        for i, x in enumerate(inds[0]):
            y = inds[1][i]
            # Add factor of 100
            tmpgraph[x, y] = weights[x]*weights[y]*tmpgraph[x, y] / 100.0
        with np.errstate(over='raise'):
            try:
                det = np.linalg.det(tmpgraph)
            except:
                det = np.linalg.det(tmpgraph/100.0)
        if 'e+' in str(det):
            safedet = str(det).split(
                'e+')[0][0:10]+'e+'+str(det).split('e+')[1]
        else:
            safedet = str(det)[0:10]
        return safedet

    def calc_suggested_spin(self, params={}):
        """calc_suggested_spin calculate the suggested spin using electron counting and given information.

        Parameters
        ----------
        params : dict, optional
            inputDict parameters , by default dict()
        """
        # Charge -> charges already assigned to components during assembly
        metals = self.find_metals()
        charge_vect = np.zeros(len(self.ase_atoms))
        # Prioritize input full charge > ase atoms charge > self.charge > metal_ox > None
        if (params.get('full_charge',None) is not None):
            charge_vect[0] = params['full_charge']
        elif (self.charge is not None) and (self.xtb_charge != self.charge):
            charge_vect[0] = self.charge
        elif np.any(self.ase_atoms.get_initial_charges() != 0): # Prioritize ASE atoms charge
            charge_vect = self.ase_atoms.get_initial_charges()
        elif self.charge is not None: 
            charge_vect[0] = self.charge
        else:
            if (params.get('metal_ox', None) is not None):
                charge_vect[0] = params['metal_ox']
            else:
                if len(metals) > 0:
                    syms = self.ase_atoms.get_chemical_symbols()
                    charge_vect[0] = np.sum([io_ptable.metal_charge_dict[syms[x]] for x in metals])
                else:
                    mol2str = self.write_mol2('cool.mol2',writestring=True)
                    tmol = io_obabel.convert_mol2_obmol(mol2str,readstring=True)
                    charge_vect[0] = tmol.GetTotalCharge()
                    
        mol_charge = np.sum(charge_vect)
        xtb_charge = copy.deepcopy(mol_charge)
        if np.any([True for x in self.ase_atoms.get_chemical_symbols() if x in io_ptable.heavy_metals]) and \
            ((params.get('metal_ox', None) is not None)):
            xtb_charge = mol_charge + (3-params.get('metal_ox', None))

        # Handle spin / magnetism
        even_odd_electrons = (np.sum([atom.number for atom in self.ase_atoms])-mol_charge) % 2
        # Prioritize full spin > metal spin > self.uhf > default metal spin
        if(params.get('full_spin', None) is None):
            if params.get('metal_spin',None) is None:
                if self.uhf is None:
                    syms = self.ase_atoms.get_chemical_symbols()
                    # If no metals -> uhf will start at 0
                    uhf = np.sum([io_ptable.metal_spin_dict[syms[x]] for x in metals])
                else:
                    uhf = self.uhf
            else:
                uhf = params['metal_spin'] # Metal spin set by io_process_input to defaults.
        else:
            uhf = params['full_spin']

        # Only correct uhf if anything is wrong with uhf -> even vs odd etc.
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

        xtb_uhf = 0
        if not np.any([True for x in self.ase_atoms.get_chemical_symbols() if x in io_ptable.heavy_metals]):
            xtb_uhf = uhf
        else: # F in core assumes for a 3+ lanthanide there are 11 valence electrons (8 once the 3+ is taken into account)
            even_odd_electrons = (np.sum([atom.number for atom in self.ase_atoms]))
            even_odd_electrons = even_odd_electrons - \
                io_ptable.elements.index(self.ase_atoms.get_chemical_symbols()[metals[0]]) + 11 - xtb_charge
            even_odd_electrons = even_odd_electrons % 2
            if (even_odd_electrons == 0):
                xtb_uhf = 0
            else:
                xtb_uhf = 1
        
        # Assign to complex
        self.charge = mol_charge
        self.uhf = uhf
        self.xtb_uhf = xtb_uhf
        self.xtb_charge = xtb_charge

    def swap_actinide(self,debug=False,skip=False):
        """swap_actinide swap actinides for lanthanides or reverse

        Parameters
        ----------
        debug : bool, optional
            print debug statements, by default False
        skip : bool, optional
            skip swapping back, by default False
        """
        if skip:
            if debug:
                print('Skipping swapping')
            pass
        elif (self.actinides_swapped) and (len(self.actinides) > 0):
            if debug:
                print('Swapping substituted lanthanides back to actinides.')
            syms = self.ase_atoms.get_chemical_symbols()
            ln_symbols = [syms[x] for x in self.actinides]
            an_symbols = [io_ptable.actinides[io_ptable.lanthanides.index(x)] for x in ln_symbols]
            for i,j in enumerate(self.actinides):
                syms[j] = an_symbols[i]
                self.atom_types[j] = an_symbols[i]
            self.actinides_swapped = False
            self.ase_atoms.set_chemical_symbols(syms)
        elif (len(self.actinides)):
            if debug:
                print('Swapping actinides to lanthanides.')
            syms = self.ase_atoms.get_chemical_symbols()
            an_symbols = [syms[x] for x in self.actinides]
            ln_symbols = [io_ptable.lanthanides[io_ptable.actinides.index(x)] for x in an_symbols]
            for i,j in enumerate(self.actinides):
                syms[j] = ln_symbols[i]
                self.atom_types[j] = ln_symbols[i]
            self.actinides_swapped = True
            self.ase_atoms.set_chemical_symbols(syms)
        else:
            if debug:
                print('No actinides present to swap.')

    def find_component_indices(self, component=0):
        """find_component_indices pull out the i'th disjoint component
        Useful for freezing atoms

        Parameters
        ----------
        component : int, optional
            index of component to extract, by default 0

        Returns
        -------
        indices, np.ndarray
            Indices of the ith component.
        """

        csg = csgraph_from_dense(self.graph)
        disjoint_components = connected_components(csg)
        indices = np.where(disjoint_components[1] == component)[0]
        return indices
    
    def calc_lig_angles_struct(self):
        """ Assign a read-in ligand structure to a specific geometry

        Returns
        -------
        userlig_dict : dict
            User ligand dictionary of angles
        denticity : int
            User ligand denticity
        """
        denticity_combinations_dict = {0:1, 1:2, 3:3, 6:4, 10:5, 15:6, 21:7, 28:8, 36:9}
        if len(self.graph) == 0:
            print('Creating imputed molecular graph! May be untrustworthy.')
            self.create_BO_dict()
        mets = self.find_metals()
        lig_angles = []
        if len(mets) == 1:
            coordats = np.nonzero(self.graph[mets[0]])[0]
            if len(coordats) == 1:
                lig_angles = []
            else:
                angs = [
                    self.ase_atoms.get_angle(x[0],mets[0],x[1]) for x in itertools.combinations(coordats,2)
                    ]
                angs = np.array(angs)[np.argsort(angs)[::-1]] # Add angles
                lig_angles += angs.tolist() # Add sorted angles as features
        else:
            print('Warning: User ligand input without metal for refernce on interatomic angles. \
                    Please pass a structure with a metal for user ligand generation.')
        lig_angles += [0.0] * (36-len(lig_angles)) # Pad with zeros
        n_ca_m_ca_angles = len(np.nonzero(lig_angles)[0])
        denticity = denticity_combinations_dict[n_ca_m_ca_angles]
        userlig_dict = {'user_lig':np.array(lig_angles)}
        return userlig_dict, denticity
    
    def classify_metal_geo_type(self,return_result=False):
        """classify_metal_geo_type calculate the actual geometry of the metal centers

        Parameters
        ----------
        return_results: bool, optional
            return the results, by default False

        Returns
        -------
        metal_center_geos : list, optional
            metal center geometries present in the mol2string.
        """
        if not len(self.graph):
            self.create_mol_graph()
        metal_inds = self.find_metals()
        geo_dict = Geometries()
        if len(metal_inds) == 0:
            raise ValueError('No metal or ind passed in this molecule.')
        elif len(metal_inds) > 1: # Look at every metal center
            metal_center_geos = []
            for metal_indx in metal_inds: 
                tmpdict = dict()
                neighs = np.nonzero(np.ravel(self.graph[metal_indx]))[0]
                if (len(neighs) < 13):
                    coord_at_positions = self.ase_atoms.positions[neighs] - self.ase_atoms.positions[metal_indx]
                    act_geo_vect = calc_all_coord_atom_angles(coord_at_positions)
                    ref_geo_labels = geo_dict.cn_geo_dict[len(neighs)]
                    ref_geos = [calc_all_coord_atom_angles(geo_dict.geometry_dict[x]) for x in ref_geo_labels]
                    mae_losses = [np.mean(np.abs(act_geo_vect - x)) for x in ref_geos] # Calc MAE loss between interatomic angles
                    sort_order = np.argsort(mae_losses)
                    m_geo_type = ref_geo_labels[np.argmin(mae_losses)]
                    tmpdict['metal'] = self.ase_atoms.get_chemical_symbols()[metal_indx]
                    tmpdict['metal_ind'] = metal_indx
                    tmpdict['metal_geo_type'] = m_geo_type
                    tmpdict['mae_angle_loss'] = mae_losses[np.argmin(mae_losses)]
                    if len(sort_order) > 1:
                        tmpdict['confidence'] = 1 - tmpdict['mae_angle_loss'] / mae_losses[sort_order[1]]
                    else:
                        tmpdict['confidence'] = 1
                    tmpdict['classification_dict'] = {ref_geo_labels[i]:mae_losses[i] for i in sort_order}
                else: 
                    tmpdict['metal'] = self.ase_atoms.get_chemical_symbols()[metal_indx]
                    tmpdict['metal_ind'] = metal_indx
                    tmpdict['metal_geo_type'] = len(neighs)
                metal_center_geos.append(tmpdict)
        else: # Just calculate geometry for one metal center.
            metal_center_geos = dict()
            metal_indx = metal_inds[0]
            neighs = np.nonzero(np.ravel(self.graph[metal_indx]))[0]
            if len(neighs) < 13:
                coord_at_positions = self.ase_atoms.positions[neighs] - self.ase_atoms.positions[metal_indx]
                act_geo_vect = calc_all_coord_atom_angles(coord_at_positions)
                ref_geo_labels = geo_dict.cn_geo_dict[len(neighs)]
                ref_geos = [calc_all_coord_atom_angles(geo_dict.geometry_dict[x]) for x in ref_geo_labels]
                mae_losses = [np.mean(np.abs(act_geo_vect - x)) for x in ref_geos]
                sort_order = np.argsort(mae_losses)
                m_geo_type = ref_geo_labels[np.argmin(mae_losses)]
                metal_center_geos['metal'] = self.ase_atoms.get_chemical_symbols()[metal_indx]
                metal_center_geos['metal_ind'] = metal_indx
                metal_center_geos['metal_geo_type'] = m_geo_type
                metal_center_geos['mae_angle_loss'] = mae_losses[np.argmin(mae_losses)]
                if len(sort_order) > 1:
                    metal_center_geos['confidence'] = (mae_losses[sort_order[1]]-metal_center_geos['mae_angle_loss'])/mae_losses[sort_order[1]]
                else:
                    metal_center_geos['confidence'] = 1
                metal_center_geos['classification_dict'] = {ref_geo_labels[i]:mae_losses[i] for i in sort_order}
            else: 
                metal_center_geos['metal'] = self.ase_atoms.get_chemical_symbols()[metal_indx]
                metal_center_geos['metal_ind'] = metal_indx
                metal_center_geos['metal_geo_type'] = len(neighs)
            metal_center_geos = [metal_center_geos]
        self.metal_center_geos = metal_center_geos
        if return_result:
            return metal_center_geos

    def get_dists(self,
                     calc_nonbonded_dists=True,
                     skin=0.3,
                     radius=None,
                     ref_ind='metals',
                     atom_pairs=None,
                     atom_type_pairs=None):
        """Calculate interatomic distances and tabulate for a given structure,
        with indices.

        Parameters
        ----------
        calc_nonbonded_dists : bool, optional
            Calulate nonbonded distances?, by default True
        skin : int, optional
            Cutoff for nonbonded distances in Angstroms (distance considered "close" by within this many angstroms
            of another coordinating atom to the metal), by default 0.3
        radius : float, optional
            Cutoff for radius of interaction to visualize around the molecule, by default None.
        ref_ind : int, optional
            Index of the atom to reference to, None will reference to all metals.
            If integer passed, the distances will be calculated from this index
            If list or array of integers passed, the distances will be calculated to all indices.
            Default 'metals'.
        atom_pairs : list(list(int)), optional
            Specific atom pairs to track, e.g. [[0,1],[1,2]] would track atoms 0-1 and 1-2 distances.
        atom_type_pairs : list(list(str)), optiona
            Types of atom pairs to track e.g. [['Fe','O'],['Fe','N']] will track all Fe-O and Fe-N systems. 

        Returns
        -------
        ml_dist_dict : dict
            Dictionary of the metal-ligand distances including indices (0-indexed)
        """
        if not len(self.graph):
            self.create_mol_graph()
        if isinstance(ref_ind,str):
            if ref_ind == 'metals':
                atoms = np.array(self.find_metals())
                metals = atoms
            else: 
                raise NotImplementedError('Not yet implemented for other keywords - only "metals".')
        elif isinstance(ref_ind,bool):
            atoms = np.array(self.find_metals())
            if len(atoms) == 0: # Organic-only
                atoms = [0] # Add single atom as index.
            metals = atoms
        elif isinstance(ref_ind,(int,float)):
            atoms = np.array([int(ref_ind)])
            metals = np.array(self.find_metals())
        elif isinstance(ref_ind,list):
            atoms = np.array(ref_ind)
            metals = np.array(self.find_metals())
        elif isinstance(ref_ind,np.ndarray):
            atoms = ref_ind
            metals = np.array(self.find_metals())
        elif ref_ind is None:
            atoms = []
            metals = np.array(self.find_metals())
        symbols = self.ase_atoms.get_chemical_symbols()
        symarray = np.array(symbols)
        ml_dist_dicts = []
        index = 0
        # Metal charges
        if (len(atoms) > 0) or (atom_pairs is not None) or (atom_type_pairs is not None):
            distmat = self.ase_atoms.get_all_distances()
            ligsmiles , _ , info_dict, = io_obabel.obmol_lig_split(
                self.write_mol2('temp.mol2',writestring=True),
                return_info=True,
                calc_all=True
                )
            if (atom_pairs is None) and (atom_type_pairs is None):
                for met in atoms:
                    con_atoms = np.nonzero(self.graph[met])[0]
                    con_atom_dists = distmat[met][con_atoms]
                    m_visited = False
                    for j,c in enumerate(con_atoms):
                        for i,ind_set in enumerate(info_dict['original_lig_inds']):
                            if c in ind_set: # Find ligand this atom belongs to.
                                ind_in_ligand = np.where(ind_set == c)[0][0]
                                ml_dist_dicts.append({
                                    'atom_pair':(met,c),
                                    'bond_type':'explicit_bond',
                                    'smiles':ligsmiles[i],
                                    'smiles_index':info_dict['mapped_smiles_inds'][i][ind_in_ligand],
                                    'distance':con_atom_dists[j],
                                    'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[met])] + \
                                                    io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                    'atom_symbols':'{}-{}'.format(symbols[met],symbols[c])
                                    })
                                index += 1
                            elif (c in metals) and (c != met) and (not m_visited):
                                m_visited=True
                                ml_dist_dicts.append({
                                    'atom_pair':(met,c),
                                    'bond_type':'explicit_bond',
                                    'smiles':None,
                                    'smiles_index':None,
                                    'distance':con_atom_dists[j],
                                    'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[met])] + \
                                                    io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                    'atom_symbols':'{}-{}'.format(symbols[met],symbols[c])
                                    })
                                index += 1
                    if calc_nonbonded_dists:
                        if radius is None:
                            other_close_atoms = np.where(distmat[met] < (np.max(con_atom_dists)+skin))[0]
                        else:
                            other_close_atoms = np.where(distmat[met] < (radius))[0]
                        other_close_atoms = np.array([x for x in other_close_atoms if x not in (con_atoms.tolist() + \
                                                    [int(met)])])
                        if len(other_close_atoms) > 0:
                            other_close_atom_dists = distmat[met][other_close_atoms]
                            for j,c in enumerate(other_close_atoms):
                                m_visited = False
                                for i,ind_set in enumerate(info_dict['original_lig_inds']):
                                    if c in ind_set: # Find ligand this atom belongs to.
                                        ind_in_ligand = np.where(ind_set == c)[0][0]
                                        ml_dist_dicts.append({
                                            'atom_pair':(met,c),
                                            'bond_type':'implicit_bond',
                                            'smiles':ligsmiles[i],
                                            'smiles_index':info_dict['mapped_smiles_inds'][i][ind_in_ligand],
                                            'distance':other_close_atom_dists[j],
                                            'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[met])] + \
                                                            io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                            'atom_symbols':'{}-{}'.format(symbols[met],symbols[c])
                                            })
                                        index += 1
                                    elif (c in metals) and (c != met) and (not m_visited):
                                        m_visited=True
                                        ml_dist_dicts.append({
                                            'atom_pair':(met,c),
                                            'bond_type':'implicit_bond',
                                            'smiles':None,
                                            'smiles_index':None,
                                            'distance':other_close_atom_dists[j],
                                            'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[met])] + \
                                                            io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                            'atom_symbols':'{}-{}'.format(symbols[met],symbols[c])
                                            })
                                        index += 1
            elif (atom_pairs is not None):
                for inds in atom_pairs:
                    i0,i1 = int(inds[0]),int(inds[1])
                    ml_dist_dicts.append({
                                'atom_pair':(i0,i1),
                                'bond_type':'explicit_ask',
                                'smiles':None,
                                'smiles_index':None,
                                'distance':distmat[i0][i1],
                                'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[i0])] + \
                                                io_ptable.rcov1[io_ptable.elements.index(symbols[i1])],
                                'atom_symbols':'{}-{}'.format(symbols[i0],symbols[i1])
                                })
            elif (atom_type_pairs is not None) and (len(metals) > 0):
                for pair_type in atom_type_pairs:
                    type1s = pair_type[0]
                    type2s = pair_type[1]
                    type1inds = np.where(symarray == type1s)[0]
                    type2inds = np.where(symarray == type2s)[0]
                    for t1 in type1inds:
                        con_atoms = np.nonzero(self.graph[t1])[0]
                        all_dists = distmat[t1]
                        for c in type2inds:
                            m_visited = False
                            if c in con_atoms:
                                for i,ind_set in enumerate(info_dict['original_lig_inds']):
                                    if c in ind_set: # Find ligand this atom belongs to.
                                        ind_in_ligand = np.where(ind_set == c)[0][0]
                                        ml_dist_dicts.append({
                                            'atom_pair':(t1,c),
                                            'bond_type':'explicit_bond',
                                            'smiles':ligsmiles[i],
                                            'smiles_index':info_dict['mapped_smiles_inds'][i][ind_in_ligand],
                                            'distance':all_dists[c],
                                            'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[t1])] + \
                                                            io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                            'atom_symbols':'{}-{}'.format(symbols[t1],symbols[c])
                                            })
                                        index += 1
                                    elif (c in metals) and (not m_visited):
                                        m_visited = True
                                        ml_dist_dicts.append({
                                            'atom_pair':(t1,c),
                                            'bond_type':'explicit_bond',
                                            'smiles':None,
                                            'smiles_index':None,
                                            'distance':all_dists[c],
                                            'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[t1])] + \
                                                            io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                            'atom_symbols':'{}-{}'.format(symbols[t1],symbols[c])
                                            })
                                        index += 1
                            else:
                                for i,ind_set in enumerate(info_dict['original_lig_inds']):
                                    if c in ind_set: # Find ligand this atom belongs to.
                                        ind_in_ligand = np.where(ind_set == c)[0][0]
                                        ml_dist_dicts.append({
                                            'atom_pair':(t1,c),
                                            'bond_type':'implicit_bond',
                                            'smiles':ligsmiles[i],
                                            'smiles_index':info_dict['mapped_smiles_inds'][i][ind_in_ligand],
                                            'distance':all_dists[c],
                                            'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[t1])] + \
                                                            io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                            'atom_symbols':'{}-{}'.format(symbols[t1],symbols[c])
                                            })
                                        index += 1
                                    elif (c in metals) and (not m_visited):
                                        m_visited = True
                                        ml_dist_dicts.append({
                                            'atom_pair':(t1,c),
                                            'bond_type':'implicit_bond',
                                            'smiles':None,
                                            'smiles_index':None,
                                            'distance':all_dists[c],
                                            'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[t1])] + \
                                                            io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                            'atom_symbols':'{}-{}'.format(symbols[t1],symbols[c])
                                            })
                                        index += 1
            elif (atom_type_pairs is not None):
                for pair_type in atom_type_pairs:
                    type1s = pair_type[0]
                    type2s = pair_type[1]
                    type1inds = np.where(symarray == type1s)[0]
                    type2inds = np.where(symarray == type2s)[0]
                    for t1 in type1inds:
                        all_dists = distmat[t1]
                        for c in type2inds:
                            if self.graph[t1][c] == 1:
                                bond_type = 'explicit_bond'
                            else:
                                bond_type = 'implicit_bond'
                            ml_dist_dicts.append({
                                'atom_pair':(t1,c),
                                'bond_type':bond_type,
                                'smiles':ligsmiles[0],
                                'smiles_index':None,
                                'distance':all_dists[c],
                                'sum_cov_radii':io_ptable.rcov1[io_ptable.elements.index(symbols[t1])] + \
                                                io_ptable.rcov1[io_ptable.elements.index(symbols[c])],
                                'atom_symbols':'{}-{}'.format(symbols[t1],symbols[c])
                                })
        else:
            raise ValueError('Need at least ref_ind, a metal, or atom_pairs, or atom_type_pairs specified!')
        df = pd.DataFrame(ml_dist_dicts)
        visited_keys = set()
        duplicates = []
        for i,row in df.iterrows():
            if tuple(sorted(row['atom_pair'])) in visited_keys:
                duplicates.append(row['atom_pair'])
            elif (row['atom_pair'][0] == row['atom_pair'][1]): # Check for identical
                duplicates.append(row['atom_pair'])
            else:
                visited_keys.add(tuple(sorted(row['atom_pair'])))
        filterdf = df[~df.atom_pair.isin(duplicates)]
        return filterdf

    def get_disjoint_molecules(self):
        """split this molecule into non-bonded sub-molecules.
        """
        if len(self.graph) < 1:
            self.create_mol_graph()
        csg = csgraph_from_dense(self.graph)
        disjoint_components = connected_components(csg)
        atom_groups = disjoint_components[1]
        unique_group_names = np.unique(atom_groups)
        mols = []
        for i in unique_group_names:
            newmol = Molecule()
            subinds = np.where(atom_groups == i)[0]
            subinds_bodict = subinds + 1
            ase_ats = self.ase_atoms[subinds]
            newbodict=dict()
            for inds,bo in self.BO_dict.items():
                ind0,ind1 = inds[0],inds[1]
                if (ind0 in subinds_bodict) and (ind1 in subinds_bodict):
                    nind0 = len(np.where(subinds_bodict < ind0)[0]) + 1
                    nind1 = len(np.where(subinds_bodict < ind1)[0]) + 1
                    newbodict[(nind0,nind1)] = bo
            natom_types = np.array(self.atom_types)[subinds].tolist()
            newmol.load_ase(ase_ats,BO_dict=newbodict,atom_types=natom_types)
            mols.append(newmol)
        return mols
    
    def detect_trans_oxos(self):
        """Detect and flag trans_oxos.

        Returns:
            trans_oxo_triples: list(tuple (3))
                List of tuples of indices of oxo1-metal-oxo2.
        """
        if len(self.graph) < 1:
            self.create_mol_graph()
        info_dict = self.split_ligs()
        oxo_inds = []
        for i,lig in enumerate(info_dict['lig_smiles']):
            if lig == '[O-2]':
                oxo_inds.append(i)
        trans_oxo_triples = []
        if isinstance(info_dict['metal_ind'],(int,float)):
            info_dict['metal_ind'] = [info_dict['metal_ind']]
        for m_ind in info_dict['metal_ind']:
            toxinds = []
            for oxind in oxo_inds:
                if m_ind in info_dict['bound_metal_inds'][oxind]:
                    toxinds.append(oxind)
            if len(toxinds) > 1:
                for pair in itertools.combinations(toxinds,2):
                    ind0 = info_dict['original_lig_inds'][pair[0]][0]
                    ind1 = info_dict['original_lig_inds'][pair[1]][0]
                    angle = self.ase_atoms.get_angle(ind0,m_ind,ind1)
                    if angle > 179:
                        trans_oxo_triples.append((ind0,m_ind,ind1))
        return trans_oxo_triples

    def functionalize_3D(self,
                         functional_groups=['carboxyllic_acid'],
                         functionalization_inds=[],
                         bond_orders=[],
                         functional_group_mol_inds=[],
                         remove_hydrogens_when_adding=[],
                         uff_opt=True,
                         xtb_opt=False,
                         core_frozen=True):
        """functionalize_3D functionalization 

        Add one to many functional groups to this molecule.

        By default NO xtb optimization will be performed.

        example:
        from architector import convert_io_molecule,view_structures
        mol = convert_io_molecule('COCC')
        mol.functionalize_3D(functional_groups=['Cl','O'],
                            functionalization_inds=[0,3],
                            bond_orders=[1,1],
                            xtb_opt=True)
        view_structures(mol)

        This should look like the 3D version of the SMILES string:
        'C(OCCO)Cl'

        Note that the bond_orders parameter is used to specify the bond order desired
        between the functional group and the molecule.

        It assumes that for each bond order added, one hydrogen is removed from the 
        molecule and one hydrogen is removed from the "functional" group added.

        This function is primarily designed for post-functionalization of
        ab-initio relaxed structures. Ideally it will save difficult-to-converge
        SCF time!

        Parameters
        ----------
        functional_groups : list[str], optional
            smiles string or name of functional_group, by default 'C'
        functionalization_inds : list, optional
            indices in the molecule where the functional_group should be added, by default [0]
        bond_orders : list, optional
            bond orders of the functional group added to the initial molecule, by default [1]
        functional_group_mol_inds : list(list(int)), optional
            which atoms in the functional group that should be bound to the metal, by default [0]
        remove_hydrogens_when_adding : list(bool), optional
            remove hydrogens when adding this functionalization?, by default True for all bonds added.
        uff_opt : bool, optional
            Relax with UFF/openbabel (turning off NOT suggested), by default True
        xtb_opt : bool, optional
            Relax with gas-phase xtb?, by default False
        core_frozen : bool, optional
            Keep the existing structure frozen, by default True
        """
        from architector.io_calc import CalcExecutor
        new_functional_groups = []
        for i,fg in enumerate(functional_groups):
            if fg in io_ptable.functional_groups_dict:
                tfg = io_ptable.functional_groups_dict[fg]
                new_functional_groups.append(tfg)
                if len(bond_orders) < i+1:
                    bond_orders.append(1)
                else:
                    bond_orders[i] = 1
                if len(functional_group_mol_inds) < i+1:
                    functional_group_mol_inds.append(0)
                else:
                    functional_group_mol_inds[i] = 0
                if len(remove_hydrogens_when_adding) < i+1:
                    remove_hydrogens_when_adding.append(True)
                else:
                    remove_hydrogens_when_adding[i] = True
            else:
                new_functional_groups.append(fg)
                if len(bond_orders) < i+1:
                    bond_orders.append(1)
                if len(functional_group_mol_inds) < i+1:
                    functional_group_mol_inds.append(0)
                if len(remove_hydrogens_when_adding) < i+1:
                    remove_hydrogens_when_adding.append(True)

        init_n_atoms = len(self.ase_atoms)
        removed_indices = []

        for i,idx in enumerate(functionalization_inds):
            if isinstance(idx,(int,float)):
                idx = int(idx) # Make sure it's an integer
                functional_group = new_functional_groups[i]
                functional_group_mol_ind = functional_group_mol_inds[i]

                removed_indices = np.array(removed_indices)
                idx = idx - len(np.where(removed_indices < idx)[0])

                funct_mol = convert_io_molecule(functional_group)

                mol_coords = self.ase_atoms.get_positions()
                mol_anums = self.ase_atoms.get_atomic_numbers()

                funct_coords = funct_mol.ase_atoms.get_positions()
                funct_anums = funct_mol.ase_atoms.get_atomic_numbers()

                funct_coords = funct_coords - funct_coords[functional_group_mol_ind]
                mol_coords = mol_coords - mol_coords[idx]

                # rotate molecule to -z
                r = Rot.align_vectors(np.array([[0.,0.,-1.]]*len(mol_coords)), mol_coords)#.reshape(1,-1))
                mol_coords = r[0].apply(mol_coords)

                self.ase_atoms.set_positions(mol_coords)

                # rotate functional group to +z 
                r = Rot.align_vectors(np.array([[0.,0.,1.]]*len(funct_coords)), funct_coords)#.reshape(1,-1))
                funct_coords = r[0].apply(funct_coords) + np.array((0.,0.,2.)) # Move to 2

                funct_mol.ase_atoms.set_positions(funct_coords)

                if remove_hydrogens_when_adding[i]: # Only if remove_hydrogens requested.

                    fg_hydrogens_inds = np.intersect1d(np.nonzero(funct_mol.graph[functional_group_mol_ind])[0],
                                                    np.where(np.array(funct_anums) == 1)[0])
                    
                    mol_hydrogens_inds = np.intersect1d(np.nonzero(self.graph[idx])[0],
                                                        np.where(np.array(mol_anums) == 1)[0])
                    
                    if (len(mol_hydrogens_inds) < bond_orders[i]):
                        raise ValueError('Molecule atom {} bound by fewer hydrogen atoms than needed!'.format(idx))
                    elif (len(fg_hydrogens_inds) < bond_orders[i]):
                        raise ValueError('Functional group atom 0 bound by fewer hydrogen atoms than needed!')
                    
                    # Distance from FG for molecule
                    mol_hydrogen_dists = np.linalg.norm(mol_coords[mol_hydrogens_inds] - np.array((0.,0.,2.)),
                                                        axis=1)
                    # Distance from molecule for FG
                    fg_hydrogen_dists = np.linalg.norm(funct_coords[fg_hydrogens_inds],
                                                        axis=1)
                    
                    mol_delete_inds = sorted(mol_hydrogens_inds[mol_hydrogen_dists.argsort()[:bond_orders[i]]])[::-1]

                    for j in mol_delete_inds:
                        self.remove_atom(j)

                    # Shift removed inds back to original index frame.
                    removed_inds = [x+len(np.where(removed_indices <= x)[0]) for x in mol_delete_inds]
                    removed_indices = removed_indices.tolist()
                    removed_indices += removed_inds

                    fg_delete_inds = sorted(fg_hydrogens_inds[fg_hydrogen_dists.argsort()[:bond_orders[i]]])[::-1]

                    for j in fg_delete_inds:
                        funct_mol.remove_atom(j)

                natoms = len(self.ase_atoms)
                newligbodict = dict()
                for key,val in funct_mol.BO_dict.items():
                    newkey = [0,0]
                    newkey[0] = natoms + int(key[0])
                    newkey[1] = natoms + int(key[1])
                    newkey = tuple(newkey)
                    newligbodict.update({newkey:val})
                newligbodict.update({(idx+1,natoms+1+functional_group_mol_ind):bond_orders[i]})

                self.ase_atoms += funct_mol.ase_atoms
                self.atom_types += funct_mol.atom_types
                if funct_mol.charge is not None:
                    self.charge += funct_mol.charge # Add charge for charged functional groups.
                    if isinstance(self.xtb_charge, (int,float)):
                        if isinstance(funct_mol.xtb_charge, (int,float)):
                            self.xtb_charge += funct_mol.xtb_charge
                        else:
                            self.xtb_charge += funct_mol.charge
                    else:
                        if isinstance(funct_mol.xtb_charge, (int,float)):
                            self.xtb_charge = funct_mol.xtb_charge
                if funct_mol.uhf is not None:
                    self.uhf += funct_mol.uhf
                    if isinstance(self.xtb_uhf, (int,float)):
                        if isinstance(funct_mol.xtb_uhf, (int,float)):
                            self.xtb_uhf += funct_mol.xtb_uhf
                    else:
                        if isinstance(funct_mol.xtb_uhf, (int,float)):
                            self.xtb_uhf = funct_mol.xtb_uhf
                self.BO_dict.update(newligbodict)
                self.create_graph_from_bo_dict()
            elif isinstance(idx,(np.ndarray,list)):
                idx = np.array([int(x) for x in idx])
                functional_group = new_functional_groups[i]
                functional_group_mol_ind = functional_group_mol_inds[i]
                remove_hydrogens = remove_hydrogens_when_adding[i]
                if not isinstance(remove_hydrogens,(list,np.ndarray)): # Recast list with default True.
                    remove_hydrogens = [True]*len(functional_group_mol_ind)

                if len(idx) != len(functional_group_mol_ind):
                    raise ValueError('Functionalization Indices must match functional_group_mol_ind in shape! \n{}!={}!'.format(
                        len(functional_group_mol_ind),len(idx)
                    ))
                
                if len(idx) != len(remove_hydrogens):
                    raise ValueError('Remove hydrogens bools must match functional_group_mol_ind in shape! \n{}!={}!'.format(
                        len(remove_hydrogens),len(idx)
                    ))

                removed_indices = np.array(removed_indices)
                for j,x in enumerate(idx):
                    idx[j] = x - len(np.where(removed_indices <= x)[0])

                funct_mol = convert_io_molecule(functional_group)

                mol_coords = self.ase_atoms.get_positions()
                mol_anums = self.ase_atoms.get_atomic_numbers()

                funct_coords = funct_mol.ase_atoms.get_positions()
                funct_anums = funct_mol.ase_atoms.get_atomic_numbers()

                funct_coords = funct_coords - np.mean(funct_coords[functional_group_mol_ind],axis=0)
                mol_coords = mol_coords - np.mean(mol_coords[idx],axis=0)

                # rotate molecule to -z
                r = Rot.align_vectors(np.array([[0.,0.,-1.]]*len(mol_coords)), mol_coords)
                mol_coords = r[0].apply(mol_coords)

                self.ase_atoms.set_positions(mol_coords)

                # rotate functional group to +z 
                r = Rot.align_vectors(np.array([[0.,0.,1.]]*len(funct_coords)), funct_coords)
                funct_coords = r[0].apply(funct_coords) + np.array((0.,0.,2.)) # Move to 2

                funct_mol.ase_atoms.set_positions(funct_coords)

                mol_delete_inds = []
                fg_delete_inds = []

                for j,fg_mol_ind in enumerate(functional_group_mol_ind):
                    if remove_hydrogens[j]: # Only if removing hydrogens requested
                        fg_hydrogens_inds = np.intersect1d(np.nonzero(funct_mol.graph[fg_mol_ind])[0],
                                                        np.where(np.array(funct_anums) == 1)[0])
                        
                        mol_hydrogens_inds = np.intersect1d(np.nonzero(self.graph[idx[j]])[0],
                                                            np.where(np.array(mol_anums) == 1)[0])
                        
                        if (len(mol_hydrogens_inds) < bond_orders[i][j]):
                            raise ValueError('Molecule atom {} bound by fewer hydrogen atoms than needed!'.format(idx))
                        elif (len(fg_hydrogens_inds) < bond_orders[i][j]):
                            raise ValueError('Functional group atom 0 bound by fewer hydrogen atoms than needed!')
                        
                        # Distance from FG for molecule
                        mol_hydrogen_dists = np.linalg.norm(mol_coords[mol_hydrogens_inds] - np.array((0.,0.,2.)),
                                                            axis=1)
                        # Distance from molecule for FG
                        fg_hydrogen_dists = np.linalg.norm(funct_coords[fg_hydrogens_inds],
                                                            axis=1)
                        
                        mol_del_inds = mol_hydrogens_inds[mol_hydrogen_dists.argsort()[:bond_orders[i][j]]].tolist()

                        if len([x for x in mol_del_inds if x in mol_delete_inds]) > 0: # Repeated deleted hydrogen
                            mol_del_inds = [x for x in mol_hydrogens_inds if x not in mol_delete_inds][:bond_orders[i][j]]
                        
                        mol_delete_inds += mol_del_inds

                        fg_del_inds = fg_hydrogens_inds[fg_hydrogen_dists.argsort()[:bond_orders[i][j]]].tolist()

                        if len([x for x in fg_del_inds if x in fg_delete_inds]) > 0: # Repeated deleted hydrogen
                            fg_del_inds = [x for x in fg_hydrogens_inds if x not in fg_delete_inds][:bond_orders[i][j]]

                        fg_delete_inds += fg_del_inds

                if np.any(remove_hydrogens):
                    mol_delete_inds = sorted(mol_delete_inds)[::-1]
                    fg_delete_inds = sorted(fg_delete_inds)[::-1]

                    for j in mol_delete_inds:
                        self.remove_atom(j)

                    # Shift removed inds back to original index frame.
                    removed_inds = [x+len(np.where(removed_indices <= x)[0]) for x in mol_delete_inds]
                    removed_indices = removed_indices.tolist()
                    removed_indices += removed_inds

                    for j in fg_delete_inds:
                        funct_mol.remove_atom(j)

                natoms = len(self.ase_atoms)
                newligbodict = dict()
                for key,val in funct_mol.BO_dict.items():
                    newkey = [0,0]
                    newkey[0] = natoms + int(key[0])
                    newkey[1] = natoms + int(key[1])
                    newkey = tuple(newkey)
                    newligbodict.update({newkey:val})
                for j,fg_mol_ind in enumerate(functional_group_mol_ind):
                    if np.any(remove_hydrogens):
                        tmpind = idx[j] - len(np.where(np.array(mol_delete_inds) <= idx[j])[0])
                    else:
                        tmpind = idx[j]
                    newligbodict.update({(tmpind+1,natoms+1+fg_mol_ind):bond_orders[i][j]})

                self.ase_atoms += funct_mol.ase_atoms
                self.atom_types += funct_mol.atom_types
                if funct_mol.charge is not None:
                    self.charge += funct_mol.charge # Add charge for charged functional groups.
                    if isinstance(self.xtb_charge, (int,float)):
                        if isinstance(funct_mol.xtb_charge, (int,float)):
                            self.xtb_charge += funct_mol.xtb_charge
                        else:
                            self.xtb_charge += funct_mol.charge
                    else:
                        if isinstance(funct_mol.xtb_charge, (int,float)):
                            self.xtb_charge = funct_mol.xtb_charge
                if funct_mol.uhf is not None:
                    self.uhf += funct_mol.uhf
                    if isinstance(self.xtb_uhf, (int,float)):
                        if isinstance(funct_mol.xtb_uhf, (int,float)):
                            self.xtb_uhf += funct_mol.xtb_uhf
                    else:
                        if isinstance(funct_mol.xtb_uhf, (int,float)):
                            self.xtb_uhf = funct_mol.xtb_uhf
                self.BO_dict.update(newligbodict)
                self.create_graph_from_bo_dict()
            else:
                raise ValueError('Unknown index {}.'.format(idx))

        freeze_indices = [x for x in range(init_n_atoms-len(removed_indices))]
        if not core_frozen:
            freeze_indices = None
        if uff_opt:
            tmpmol = self.write_mol2('temp.mol2',writestring=True)
            obmol_opt = CalcExecutor(tmpmol, relax=True, method='UFF', fix_indices=freeze_indices,
                                     trans_oxo_triples=self.detect_trans_oxos())
            self.ase_atoms.set_positions(obmol_opt.mol.ase_atoms.get_positions())
        if xtb_opt:
            tmpmol = self.write_mol2('temp.mol2',writestring=True)
            obmol_opt = CalcExecutor(tmpmol, relax=True,
                                     method='GFN2-xTB',
                                     fix_indices=freeze_indices,
                                     trans_oxo_triples=self.detect_trans_oxos()
                                     )
            self.ase_atoms.set_positions(obmol_opt.mol.ase_atoms.get_positions())


    def split_ligs(self):
        """Split the molecule into ligands with labelled information.

        Returns
        -------
        info_dict : dict
            Dictionary with all ligand information.
        """
        ligsmiles , catoms , info_dict, = io_obabel.obmol_lig_split(
            self.write_mol2('temp.mol2',writestring=True),
            return_info=True,
            calc_all=True
            )
        info_dict['lig_smiles'] = ligsmiles
        info_dict['lig_metal_coordatoms'] = catoms
        return info_dict