""" 
Code containing molecule/io
Molecule class encodes graph/bond order information and can handle a variety of types of inputs.
convert_io_molecule handles conversion to molecule object.

Also contains sanity check and graph sanity check routines for checking final geometries.

Developed by Michael Taylor
"""

import ase
from ase.io import read
from ase.atoms import Atoms
from ase.atom import Atom
import numpy as np
import re
import copy
import itertools
import architector
from architector import io_obabel
import architector.io_ptable as io_ptable
from io import StringIO
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
        elif isinstance(structure,str) and (len(structure.split('\n')) > 3) and (structure.split('\n')[0].replace(' ','').isnumeric()) \
            and ('FORCES' in structure) and ('ENERGY' in structure): # RXYZ string.
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

    def __init__(self, in_ase=False, BO_dict={}, atom_types=[], cell=[], charge=None,
                uhf=None, xtb_uhf=None, xtb_charge=None):
        self.dists_sane = True
        self.sanity_check_dict = {}
        if isinstance(in_ase,ase.atoms.Atoms):
            self.ase_atoms = in_ase.copy()
            self.BO_dict = BO_dict
            self.charge = charge
            self.uhf = uhf
            self.xtb_uhf = xtb_uhf
            self.xtb_charge = xtb_charge
            self.atom_types = atom_types
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
        elif isinstance(in_ase,bool):
            self.ase_atoms = None
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
        else:
            raise ValueError('Detect_charge_spin not implemented for non-metal containing systems.')
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

    def find_metal(self):
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
            print('Assigning first metal as metalind.')
            metalind = metalinds[0]
        else:
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
        metals = [i for i,x in enumerate(self.ase_atoms) if (x.symbol in io_ptable.all_metals)]
        return metals

    def remove_atom(self,ind):
        """remove_atom delete an atom from the molecule with ind

        Parameters
        ----------
        ind : int
            index to remove
        """
        del self.ase_atoms[ind]
        self.graph = np.delete(np.delete(self.graph,ind,0),ind,1)
        del self.atom_types[ind]
        BO_dict_tmp = {x:y for x,y in self.BO_dict.items() if (ind+1 not in x)}
        bo_dict_out = dict()
        for x,y in BO_dict_tmp.items():
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

    def append_ligand(self, ligand=None):
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
        natoms = len(self.ase_atoms)
        newligbodict = dict()
        for key,val in lig_bo_dict.items():
            newkey = [0,0]
            if int(key[0]) > 1:
                newkey[0] = natoms + int(key[0]) - 1
            else:
                newkey[0] = 1
            if int(key[1]) > 1:
                newkey[1] = int(key[1]) + natoms - 1
            else:
                newkey[1] = 1
            newkey = tuple(newkey)
            newligbodict.update({newkey:val})
        self.BO_dict.update(newligbodict)
        lcs = lig_ase_atoms.get_initial_charges().sum()
        self.charge += lcs
        self.xtb_charge += lcs
        self.ase_atoms += lig_ase_atoms
        self.atom_types += lig_atom_types
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
            run_check = params['full_sanity_checks']
            factor = params['full_graph_sanity_cutoff']
        elif (len(params) > 0):
            run_check = params['assemble_sanity_checks']
            factor = params['assemble_graph_sanity_cutoff']
        else:
            params = {'covrad_metal':None}
        sane = self.dists_sane
        graph_dists_dict = {}
        if run_check:
            if len(self.ase_atoms) > 1: # Don't test distances for single atom
                posits = self.ase_atoms.get_positions()
                m_inds = [i for i,x in enumerate(self.ase_atoms.get_chemical_symbols()) if x in io_ptable.all_metals]
                m_ind=None
                if len(m_inds) > 1:
                    raise ValueError('Sanity check only implemented for mononuclear so far.')
                elif len(m_inds) == 1:
                    m_ind = m_inds[0]
                mrad = None
                if isinstance(params['covrad_metal'],float):
                    mrad = params['covrad_metal']
                if np.any(np.isnan(posits)): # Any nan in positions
                    sane = False
                    graph_dists_dict.update({'containsNAN_posit':True})
                else:
                    all_dists = self.ase_atoms.get_all_distances()
                    cov_radii = np.array([io_ptable.rcov1[x] for x in self.ase_atoms.get_atomic_numbers()]) 
                    if (not (m_ind is None)) and (not (mrad is None)):
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
        if (len(params) > 0)  and (not assembly):
            run_check = params['full_sanity_checks']
            smallest_dist_cutoff = params['full_smallest_dist_cutoff']
            min_dist_cutoff = params['full_min_dist_cutoff']
        elif (len(params) > 0) :
            run_check = params['assemble_sanity_checks']
            smallest_dist_cutoff = params['assemble_smallest_dist_cutoff']
            min_dist_cutoff = params['assemble_min_dist_cutoff']
        sane = self.dists_sane
        min_dist_dict = {}
        smallest_dist_dict = {}
        if run_check:
            atoms = self.ase_atoms.copy()
            if len(atoms) > 1: # Don't test distances for single atom
                posits = atoms.get_positions()
                m_inds = [i for i,x in enumerate(atoms.get_chemical_symbols()) if x in io_ptable.all_metals]
                m_ind=None
                if len(m_inds) > 1:
                    raise ValueError('Sanity check only implemented for mononuclear so far.')
                elif len(m_inds) == 1:
                    m_ind = m_inds[0]
                mrad = None
                if isinstance(covrad_metal,float):
                    mrad = covrad_metal
                elif isinstance(params,dict):
                    if isinstance(params.get('covrad_metal',False),float):
                        mrad = params['covrad_metal']
                if np.any(np.isnan(posits)): # Any nan in positions
                    if params.get('debug',False):
                        print('Nan in positions.')
                    sane = False
                else:
                    all_dists = atoms.get_all_distances()
                    cov_radii = np.array([io_ptable.rcov1[x] for x in atoms.get_atomic_numbers()])
                    if (not (m_ind is None)) and (not (mrad is None)):
                        cov_radii[m_ind] = mrad
                    for i in range(0,len(atoms)):
                        j_list = list(range(0,len(atoms)))
                        j_list.remove(i)
                        i_dists = []
                        for j in j_list:
                            i_dists.append(all_dists[i,j])
                            if all_dists[i,j] < smallest_dist_cutoff*(cov_radii[i] + cov_radii[j]): # Check for extra crowded metals
                                sane = False
                                smallest_dist_dict.update({'Cutoff':smallest_dist_cutoff})
                                smallest_dist_dict.update({(i,j):all_dists[i,j]/(cov_radii[i] + cov_radii[j])})
                                if params.get('debug',False):
                                    print('Dist short: ', all_dists[i,j])
                        if min(i_dists) > min_dist_cutoff: # Catch cases where atom shot off metal center or blown up structure
                            sane = False
                            min_dist_dict.update({'Cutoff':min_dist_cutoff})
                            min_dist_dict.update({i:min(i_dists)})
                            if params.get('debug',False):
                                print('Mindist long: ', min(i_dists))
        self.dists_sane = sane
        self.sanity_check_dict.update({'Smallest_Dist_Checks':smallest_dist_dict, 'Min_Dist_Checks':min_dist_dict})

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

    def calc_suggested_spin(self,params={}):
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
