"""
IO module for open babel.

Developed by Dan Burril and Michael Taylor
"""

# Imports
import ase
import numpy as np
import architector
import architector.io_ptable as io_ptable
import warnings

from ase.io import read
from ase import units
from openbabel import openbabel as ob
from openbabel import pybel
from io import StringIO
from scipy.sparse import csgraph
import numpy as np
from pynauty import Graph as pnGraph
from pynauty import canon_label

ob_log_handler = ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0) # Set warnings to only critical.

warnings.filterwarnings('ignore') # Supress warnings.

# Functions
def smiles2xyz(smilesStr,addHydrogens=True):
    '''smiles2xyz
    Convert smiles string to xyz string.

    smilesStr : str
        smiles to convert to xyz file
    addHydrogens : bool
        whether to add hydrogens or not?!?, default True
    '''

    # Set up conversion
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("smi", "xyz")
    mol = ob.OBMol()
    obConversion.ReadString(mol, smilesStr)

    # Add hydrogens
    if (addHydrogens == True):
        mol.AddHydrogens()

    # Generate 3d structure
    builder = ob.OBBuilder()
    builder.Build(mol)

    # Set up force field
    FF = ob.OBForceField.FindForceField("MMFF94")
    FF.Setup(mol)

    # Optimize energy
    FF.ConjugateGradients(2000,1e-6)
    FF.GetCoordinates(mol)

    # Perform conversion
    obConversion.Convert()

    return obConversion.WriteString(mol).strip()

def smiles2Atoms(smilesStr,addHydrogens=True):
    """smiles2Atoms
    Convert SMILES string to Atoms object.

    Parameters
    ----------
    smilesStr : str
        smiles string
    addHydrogens : bool, optional
       add hydrogens to the molecule?, by default True

    Returns
    -------
    ats : ase.atoms.Atoms
        ase atoms of the passed smiles string string
    """

    # Variables
    symList = []        # Hold atomic symbols
    posList = []        # Hold atomic positions

    # Convert smiles to xyz
    xyzStr = smiles2xyz(smilesStr,addHydrogens=addHydrogens).split("\n")[2:]

    # Get symbols and positions
    for line in xyzStr:
        # Format line
        line = line.strip().split()

        # Get information
        symList.append(line[0])
        pos = [float(val) for val in line[1:]]
        posList.append(pos) 

    # Create Atoms object
    ats = ase.Atoms("".join(symList),positions=posList)

    # Center at COM
    ats.translate(-ats.get_center_of_mass())
    return ats

def get_OBMol_coords_anums_graph(OBMol, return_coords=True, get_types=False):
    """get_OBMol_coords_anums_graph 
    Mine obmol structure for molecular graph, atomic numbers, and coordinates.

    Parameters
    ----------
    OBMol : OBMol
        OBMol object with information
    return_coords : bool, optional
        Return coordinates, default True
    get_types : bool, optional
        Return atom types instead of numbers, default False

    Returns
    -------
    coords : np.ndarray
        NX3 array of xyz coordinates
    anums : np.ndarray
        N array of atomic numbers
    outgraph : np.ndarray
        NXN array of molecular graph
    """
    
    anums = []
    for i, atom in enumerate(ob.OBMolAtomIter(OBMol)):
        if not get_types:
            anums.append(atom.GetAtomicNum())
        else:
            anums.append(atom.GetType())
    coords = []
    if return_coords:
        for i, atom in enumerate(ob.OBMolAtomIter(OBMol)):
            vect = (atom.GetX(),atom.GetY(),atom.GetZ())
            coords.append(vect)
    outgraph = np.zeros((len(anums),len(anums)))
    for obbond in ob.OBMolBondIter(OBMol):
        outgraph[obbond.GetBeginAtomIdx()-1,obbond.GetEndAtomIdx()-1] = 1
        outgraph[obbond.GetEndAtomIdx()-1,obbond.GetBeginAtomIdx()-1] = 1
    coords = np.array(coords)
    return coords, anums, outgraph

def get_OBMol_bo_dict_atom_types(OBMol,metal_passed=True):
    """get_OBMol_coords_anums_graph 
    Mine obmol structure for BO dict and atom types

    Parameters
    ----------
    OBMol : OBMol
        OBMol object with information

    Returns
    -------
    bo_dict : dict
        Bond order dictionary with {(i,j):order} format
        Bond i,j 1-indexed with 1 always corresponding to the metal atom.
    atypes : list
        Atom types of the non-metal atoms
    """
    
    atypes = []
    ttab = ob.OBTypeTable()
    ttab.SetFromType('INT')
    ttab.SetToType('SYB')
    natoms = OBMol.NumAtoms()
    for i, atom in enumerate(ob.OBMolAtomIter(OBMol)):
        atypes.append(ttab.Translate(atom.GetType()))
    atypes = atypes[:-1] # Get rid of last atom (will always be metal)
    bo_dict = dict()
    for obbond in ob.OBMolBondIter(OBMol):
        bo = obbond.GetBondOrder()
        i = obbond.GetBeginAtomIdx()
        j = obbond.GetEndAtomIdx()
        if metal_passed:
            if i == natoms:
                i = 1
            else:
                i = 1 + i
            if j == natoms:
                j = 1
            else:
                j = 1 + j
            if i > j:
                bo_dict.update({(j,i):bo})
            else:
                bo_dict.update({(i,j):bo})
        else:
            if i > j:
                bo_dict.update({(j,i):bo})
            else:
                bo_dict.update({(i,j):bo})
    return bo_dict, atypes

def check_mmff_okay(OBMol):
    """check_mmff_okay 
    check if mmff94 applicable to chemistry

    Parameters
    ----------
    OBMol : OBmol
        openbabel of the molecule

    Returns
    -------
    mmff94good : bool
        whether this structure is okay to evaluate with mmff94.
    """
    mmff94_good_elements = ['C','H','N','O','F','Si','P','S','Br', 'Cl', 'I']
    mmff94_good_anums = [io_ptable.elements.index(x) for x in mmff94_good_elements]
    mmff94good = True
    for i, atom in enumerate(ob.OBMolAtomIter(OBMol)):
        if atom.GetAtomicNum() not in mmff94_good_anums:
            mmff94good = False
    return mmff94good
            

def Neutralize(OBmol):
    """Neutralize 
    apply openbabel neutralization procedure
    Useful for binding energies.

    Parameters
    ----------
    OBmol : openbabel.OBmol
       molecule to neutralize

    Returns
    -------
    changed : bool 
        wheter or not the molecule was changed
    """
    neutralize = ob.OBOp.FindType("neutralize")
    option = ""
    changed = neutralize.Do(OBmol, option)
    return changed


def get_obmol_smiles(smilesStr,
                    addHydrogens=True,
                    neutralize=False,
                    build=True,
                    functionalizations=None):
    """get_obmol_smiles
    convert smiles to OBmol instance

    Parameters
    ----------
    smilesStr : str
        smiles string
    addHydrogens : bool, optional
        Use add hydrogens function, by default True
    neutralize : bool, optional
        Whether to add/remove protons to make a neutral ligand, default False
    build : bool, optional
        whether to build into 3D, default True
    functionalizations : list, optional
        list of functionalization dictionaries, default None

    Returns
    -------
    OBmol : OBMol
        Built openbabel molecule
    """
    # Set up conversion
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("smi", "xyz")
    OBmol = ob.OBMol()
    obConversion.ReadString(OBmol, smilesStr)

    if (functionalizations is not None):
        for fg in functionalizations:
            OBmol = functionalize(OBmol, 
                                  functional_group=fg['functional_group'],
                                  smiles_inds=fg['smiles_inds'])

    if neutralize:
        _ = Neutralize(OBmol)

    # Add hydrogens
    if (addHydrogens == True):
        OBmol.AddHydrogens()

    if build:
        OBmol = build_3D(OBmol,addHydrogens=addHydrogens)
        return OBmol
    else:
        return OBmol


def get_smiles_obmol(OBmol,canonicalize=False):
    """get_obmol_smiles
    convert smiles to OBmol instance

    Parameters
    ----------
    OBmol : ob.OBMol
        Built openbabel molecule
    canonicalize : bool, optional
        canonicalize the smiles?
    
    Returns
    -------
    smiles : str
        smiles of the OBmol
    """
    # Set up conversion
    obConversion = ob.OBConversion()
    obConversion.SetOutFormat('smi')
    if canonicalize:
        obConversion.SetOutFormat('can')
    smiles = obConversion.WriteString(OBmol).split()[0]
    return smiles


def canonicalize_smiles(insmiles):
    """get_obmol_smiles
    convert smiles to OBmol instance

    Parameters
    ----------
    insmiles : str
        smiles of the moleucle
    
    Returns
    -------
    can_smiles : str
       canonicalized smiles of the molecule
    """
    # Set up conversion
    obConversion = ob.OBConversion()
    obConversion.SetInFormat('smi')
    obConversion.SetOutFormat('can')
    OBmol = ob.OBMol()
    obConversion.ReadString(OBmol, insmiles)
    can_smiles = obConversion.WriteString(OBmol).split()[0]
    return can_smiles


def build_3D(OBmol,addHydrogens=True):
    """build_3D take a 2D OBmol structure and build to 3D

    Parameters
    ----------
    OBmol : ob.OBMol
        2D OBmol structure
    addHydrogens : bool, optional
        whether to add hydrogens before building, by default True

    Returns
    -------
    OBmol
        3D Built OBmol structure
    """
    # Generate 3d structure
    if (addHydrogens == True):
        OBmol.AddHydrogens()
    builder = ob.OBBuilder()
    builder.Build(OBmol)

    mmff94_ok = check_mmff_okay(OBmol)

    # Set up force field
    if mmff94_ok:
        FF = ob.OBForceField.FindForceField("mmff94")
    else:
        FF = ob.OBForceField.FindForceField('UFF')
    FF.Setup(OBmol)

    # Optimize energy
    FF.ConjugateGradients(2000,1e-6)
    FF.GetCoordinates(OBmol)
    return OBmol


def generate_obmol_conformers(smiles, rmsd_cutoff=0.4, conf_cutoff=3000, energy_cutoff=50.0, 
        confab_verbose = False, output_format='mol2', neutralize=False, functionalizations=None):
    """generate_obmol_conformers 
    generate conformers with openbabel for given smiles
    using confab conformer generation routine
    O'Boyle NM, Vandermeersch T, Flynn CJ, Maguire AR, Hutchison GR. Confab 
    - Systematic generation of diverse low-energy conformers. Journal of Cheminformatics. 
    2011;3:8. doi:10.1186/1758-2946-3-8.

    Parameters
    ----------
    smiles : str
        smiles to generate conformers 
    rmsd_cutoff : float, optional
        cutoff for how similar conformers, by default 0.4
    conf_cutoff : int, optional
        total number of conformers to generate, by default 3000
    energy_cutoff : float, optional
        how similar in energy, by default 50.0
    confab_verbose : bool, optional
        give more detailed output, by default False
    output_format : str, optional
        which format to output , by default 'mol2'
    neutralize : bool, optional
        neutralize smiles?, by default False
    functionalizations : dict, optional
        add functionalizations?, by default None

    Returns
    -------
    output_strings : list (str)
        list of conformers generated as whatever format desired
    """
    obmol = get_obmol_smiles(smiles,
                             neutralize=neutralize,
                             functionalizations=functionalizations)
    mmff94_ok = check_mmff_okay(obmol)
    if mmff94_ok:
        FF = ob.OBForceField.FindForceField("MMFF94")
        FF.Setup(obmol) # Make sure setup works OK
    else:
        FF = ob.OBForceField.FindForceField("MMFF94")
        FF.Setup(obmol) # Make sure setup works OK
    FF.DiverseConfGen(rmsd_cutoff, conf_cutoff, energy_cutoff, confab_verbose)
    FF.GetConformers(obmol)
    confs_to_write = obmol.NumConformers()
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat(output_format)
    output_strings = []
    for conf_num in range(confs_to_write):
        obmol.SetConformer(conf_num)
        output_strings.append(obconversion.WriteString(obmol))
    return output_strings


def functionalize(OBmol,functional_group='C',smiles_inds=[0]):
    """functionalize functionalization routine

    Parameters
    ----------
    OBmol : ob.OBMol
        Un"built" 3D ligand
    functional_group : str, optional
        smiles string or name of functional_group, by default 'C'
    smiles_inds : list, optional
        indices where the functional_group should be added, by default [0]

    Returns
    -------
    OBmol : ob.OBMol
        functionalized (unbuilt 3D ligand)
    """

    if functional_group in io_ptable.functional_groups_dict:
        functional_group = io_ptable.functional_groups_dict[functional_group]

    second_mol = get_obmol_smiles(functional_group,build=False,neutralize=False,addHydrogens=False)

    for idx in smiles_inds:
        start_index = OBmol.NumAtoms()
        for i,atom in enumerate(ob.OBMolAtomIter(second_mol)):
            OBmol.AddAtom(atom)
            
        for obbond in ob.OBMolBondIter(second_mol):
            OBmol.AddBond(obbond.GetBeginAtomIdx()+start_index,obbond.GetEndAtomIdx()+start_index,obbond.GetBondOrder())
            
        OBmol.AddBond(start_index+1,idx+1,1)
        for i,atom in enumerate(ob.OBMolAtomIter(OBmol)):
            if (i == start_index) or (i == idx):
                atom.SetImplicitHCount(atom.GetImplicitHCount()-1)

    return OBmol

def convert_obmol_ase(OBMol,posits=None,set_zero=False,add_hydrogens=False):
    """convert_obmol_ase 
    convert obmol to ase

    Parameters
    ----------
    OBMol : OBmol
        Obmol to convert to ASE atoms
    posits : np.ndarray, optional
        positions, by default None
    set_zero : bool, optional
        set positions to origin
    add_hydrogens :bool, optional
        add hydrogens before output, default False

    Returns
    -------
    ase_atoms : ase.Atoms
        converted OBmol
    """
    if hasattr(posits,'__len__'):
        last_atom_index = OBMol.NumAtoms()
        # Set metal to zero
        metal_coords = (posits[last_atom_index-1,0],posits[last_atom_index-1,1],posits[last_atom_index-1,2])
        # set coordinates using OBMol to keep bonding info
        for i, atom in enumerate(ob.OBMolAtomIter(OBMol)):
            atom.SetVector(posits[i, 0]-metal_coords[0], posits[i, 1]-metal_coords[1], posits[i, 2]-metal_coords[2])
    if set_zero:
        for i, atom in enumerate(ob.OBMolAtomIter(OBMol)):
            atom.SetVector(0.0, 0.0, 0.0)
    
    if add_hydrogens:
        OBMol.AddHydrogens()
        mmff94_ok = check_mmff_okay(OBMol)

        # Set up force field
        if mmff94_ok:
            FF = ob.OBForceField.FindForceField("mmff94")
        else:
            FF = ob.OBForceField.FindForceField('UFF')
        FF.Setup(OBMol)

        # Optimize energy
        FF.ConjugateGradients(2000,1e-6)
        FF.GetCoordinates(OBMol)

    # Convert to ASE
    obConversion = ob.OBConversion()
    obConversion.SetOutFormat('xyz')
    xyzStr = obConversion.WriteString(OBMol).strip()
    f = StringIO(xyzStr)
    ase_atoms = read(f,format='xyz',parallel=False)
    return ase_atoms


def convert_ase_obmol(ase_atoms):
    """convert_obmol_ase 
    convert obmol to ase

    Parameters
    ----------
    ase_atoms : ase.Atoms
        structure to convert to openbabel

    Returns
    -------
    OBMol : ob.OBMol
        converted ASE atoms
    """

    OBMol = ob.OBMol()
    for atom in ase_atoms:
        obatom = ob.OBAtom()
        obatom.SetAtomicNum(int(atom.number))
        obatom.SetVector(atom.x,atom.y,atom.z)
        OBMol.AddAtom(obatom)
    return OBMol


def obmol_opt(structure,center_metal=False,fix_m_neighbors=True,
              return_energy=False):
    """obmol_opt take in a structure and optimize with openbabel
    return as ase atoms as default
    Will default to MMFF94 if it is applicable - otherwise it is UFF.

    Parameters
    ----------
    structure : ase.atoms.Atoms
        Structure to optimize with UFF / mmff94
    center_metal : bool, optional
        Move the metal to (0,0,0), default False
    fix_m_neighbors : bool, optional
        Fix the metal neighbors during optimization?, default False
    return_energy : bool, optional
        Return the energy of the optimized structure?, default False

    Returns
    ----------
    out_atoms : ase.atoms.Atoms
        structure
    energy : float, optional
        energy of the structure in UFF / MMFF94 if applicable.
    """
    if isinstance(structure,ase.atoms.Atoms):
        OBMol = convert_ase_obmol(structure)
    elif isinstance(structure,str):
        if 'TRIPOS' in structure:
            OBMol = convert_mol2_obmol(structure,readstring=True)
        elif structure[-5:] == '.mol2':
            OBMol = convert_mol2_obmol(structure,readstring=False)
    elif isinstance(structure,architector.io_molecule.Molecule):
        mol2str = structure.write_mol2('cool.mol2', writestring=True)
        OBMol = convert_mol2_obmol(mol2str, readstring=True)

    mmff94_ok = check_mmff_okay(OBMol)

    # Set up force field
    if mmff94_ok:
        FF = ob.OBForceField.FindForceField("mmff94")
    else:
        FF = ob.OBForceField.FindForceField('UFF')
    
    if fix_m_neighbors:
        _,anums,graph = get_OBMol_coords_anums_graph(OBMol, return_coords=False, get_types=False)
        syms = [io_ptable.elements[x] for x in anums]
        mets = [i for i,x in enumerate(syms) if x in io_ptable.all_metals]
        if len(mets) == 1: # Freeze metal and neighbor positions - relax ligands
            frozen_atoms = [mets[0]+1] + (np.nonzero(np.ravel(graph[mets[0]]))[0] + 1).tolist()
            constr = ob.OBFFConstraints()
            for j in frozen_atoms:
                constr.AddAtomConstraint(int(j))
        elif len(mets) > 1:
            constr = ob.OBFFConstraints()
            print('Warning : Multiple Metals present for FF optimization.')
        elif len(mets) == 0:
            constr = ob.OBFFConstraints()
            # print('No Metals present for FF optimization.')
        FF.Setup(OBMol,constr)
    else:
        FF.Setup(OBMol)

    # Optimize energy
    FF.ConjugateGradients(2000,1e-6)
    FF.GetCoordinates(OBMol)
    energy = FF.Energy()
    if mmff94_ok:
        energy = energy * units.kcal / units.mol
    else:
        energy = energy * units.kJ / units.mol
    out_atoms = convert_obmol_ase(OBMol)
    if center_metal:
        m_ind = [i for i,x in enumerate(out_atoms.get_chemical_symbols()) if x in io_ptable.all_metals]
        if len(m_ind)  == 1:
            new_posits = out_atoms.get_positions()-out_atoms.get_positions()[m_ind]
            out_atoms.set_positions(new_posits)
    if return_energy:
        return out_atoms,energy
    else:
        return out_atoms


def obmol_energy(structure):
    """obmol_energy take in a structure and use openbabel
    to evaluate energy with UFF/MMFF94.

    Parameters
    ----------
    structure : ase.atoms.Atoms
        UFF / mmff94 optimized structure.
    """
    if isinstance(structure,ase.atoms.Atoms):
        OBMol = convert_ase_obmol(structure)
    elif isinstance(structure,str):
        if 'TRIPOS' in structure:
            OBMol = convert_mol2_obmol(structure,readstring=True)
        elif structure[-5:] == '.mol2':
            OBMol = convert_mol2_obmol(structure,readstring=False)
    elif isinstance(structure,architector.io_molecule.Molecule):
        mol2str = structure.write_mol2('cool.mol2', writestring=True)
        OBMol = convert_mol2_obmol(mol2str, readstring=True)

    mmff94_ok = check_mmff_okay(OBMol)

    # Set up force field
    if mmff94_ok:
        FF = ob.OBForceField.FindForceField("mmff94")
    else:
        FF = ob.OBForceField.FindForceField('UFF')

    energy = FF.Energy()
    if mmff94_ok: # Convert to eV
        energy = energy * units.kcal / units.mol
    else:
        energy = energy * units.kJ / units.mol
    # print(FF.GetUnit()) -> Get units for other FFs potentially
    return energy    


def add_dummy_metal(Conf3D,coordList):
    """add_dummy_metal add a dummy metal to a conformer

    Parameters
    ----------
    Conf3D : ob.OBmol
        ligand conformer, likely built in 2D
    coordList : list
        coordination sites of the ligand to the metal
    """
    dummy_metal = ob.OBAtom() # Add the dummy metal to the OBmol
    dummy_metal.SetAtomicNum(26) # Add arbitrary dummy metal - Fe for now - will be removed later
    ttab = ob.OBTypeTable() # Reset types for replication purposes
    ttab.SetFromType('INT')
    ttab.SetToType('INT')
    atypes_old = []
    for i, atom in enumerate(ob.OBMolAtomIter(Conf3D)):
        atypes_old.append(ttab.Translate(atom.GetType()))
    atypes_old.append('Fe')
    Conf3D.AddAtom(dummy_metal)
    for i, atom in enumerate(ob.OBMolAtomIter(Conf3D)):
        atom.SetType(atypes_old[i])
    for i in coordList:
        Conf3D.AddBond(int(i+1), Conf3D.NumAtoms(), 1)


def convert_mol2_obmol(mol2,readstring=True):
    """convert_mol2_obmol
    mol2 to OBMol

    Parameters
    ----------
    mol2 : str
        either filename or mol2 string
    readstring : bool, optional
        read from string or from file, by default True

    Returns
    -------
    obmol : ob.OBMol
        openbabel of the mol2 file
    """
    conv = ob.OBConversion()
    obmol = ob.OBMol()
    conv.SetInFormat('mol2')
    if readstring:
        conv.ReadString(obmol,mol2)
    else:
        conv.ReadFile(obmol,mol2)
    return obmol


def convert_obmol_mol2(OBmol):
    """convert_obmol_mol2
    OBmol to mol2 string

    Parameters
    ----------
    OBmol : ob.OBMol
        OBMol object

    Returns
    -------
    mol2str : str
        mol2 str of the OBmol 
    """
    conv = ob.OBConversion()
    conv.SetOutFormat('mol2')
    mol2str = conv.WriteString(OBmol)
    return mol2str


def convert_xyz_obmol(xyz,readstring=True):
    """convert_xyz_obmol
    xyz to OBMol

    Parameters
    ----------
    xyz : str
        either filename or xyz string
    readstring : bool, optional
        read from string or from file, by default True

    Returns
    -------
    obmol : ob.OBMol
        openbabel of the xyz file
    """
    conv = ob.OBConversion()
    obmol = ob.OBMol()
    conv.SetInFormat('xyz')
    if readstring:
        conv.ReadString(obmol,xyz)
    else:
        conv.ReadFile(obmol,xyz)
    return obmol


def remove_obmol_metals(Conf3D):
    """remove_obmol_metals remove metals from obmol

    Parameters
    ----------
    Conf3D : ob.OBMol
        openbabel molecule with a metal atom

    Returns
    -------
    Conf3D : ob.OBMol
        moleucle with no metal atoms remaining
    """
    rem_list = []
    for i, atom in enumerate(ob.OBMolAtomIter(Conf3D)):
        anum = atom.GetAtomicNum()
        if io_ptable.elements[anum] in io_ptable.all_metals:
            rem_list.append(i+1)
    rem_list = sorted(rem_list)[::-1]
    for ind in rem_list:
        metal_atom = Conf3D.GetAtom(ind)
        Conf3D.DeleteAtom(metal_atom)
    return Conf3D


def obmol_lig_split(mol2string,return_info=False,calc_coord_atoms=True):
    """obmol_lig_split 
    Take in a mol2string and use openbabel to split into ligands, convert to smiles, 
    and calculate metal-ligand coordinating atoms implicit in the mol2string.

    Parameters
    ----------
    mol2string : str
       mol2string of ideally mononuclear metal complex.
    return_info : bool, optional
        return information dictionary as well with metal,ligand_charges,coordAt symbols, default False
    calc_coord_atoms : bool, optional
        return the coordination atoms for smiles, default True

    Returns
    -------
    ligand_smiles : list(str)
        list of the ligand smiles present in the complex
    coord_atom_lists : list(list)
        list of coordinating atom indices for the smiles str
    """
    obmol = convert_mol2_obmol(mol2string)
    _,anums,graph = get_OBMol_coords_anums_graph(obmol)
    bo_dict, _ = get_OBMol_bo_dict_atom_types(obmol,metal_passed=False)
    met_inds = [i for i,x in enumerate(anums) if (io_ptable.elements[x] in io_ptable.all_metals)]
    shape = graph.shape
    only_mets_graph = np.zeros(shape)
    init_graph = graph.copy()
    # Create 2 graphs - one with metals removed, one with only the metals graphs
    for ind in sorted(met_inds):
        only_mets_graph[:,ind] = graph[ind]
        only_mets_graph[ind,:] = graph[ind]
        # Zero out the deleted inds
        init_graph[ind,:] = np.zeros(shape[0])
        init_graph[:,ind] = np.zeros(shape[0])
    csg = csgraph.csgraph_from_dense(init_graph)
    # Break apart zeroed graph into connected components
    disjoint_components = csgraph.connected_components(csg)[1]
    ligs_inds = []
    for ind in sorted(list(set(disjoint_components))): # sort for reproducability
        subgraph = np.where(disjoint_components == ind)[0]
        sg = np.array([x for x in subgraph if x not in met_inds]) # Check not deleted atoms
        sg.sort()
        if len(sg) > 0:
            ligs_inds.append(sg)
    ligand_smiles = []
    coord_atom_lists = []
    for lig in ligs_inds:
        lig = lig.tolist()
        ligobmol = ob.OBMol()
        coord_atom_list = []
        for i,atom_ind in enumerate(lig):
            atom_ind += 1
            for l, atom in enumerate(ob.OBMolAtomIter(obmol)):
                if (l+1 == atom_ind): 
                    ligobmol.AddAtom(atom)
            for k in bo_dict.keys():
                if (atom_ind in k):
                    other_ind = [x for x in k if x!=atom_ind][0] - 1
                    if other_ind in met_inds:
                        coord_atom_list.append(i)
        for k in bo_dict.keys():
            if (k[0]-1 in lig) and (k[1]-1 in lig):
                start_ind = lig.index(k[0]-1) + 1
                end_ind = lig.index(k[1]-1) + 1
                ligobmol.AddBond(start_ind,end_ind,bo_dict[k])
        ligobmol.PerceiveBondOrders()
        # Key block for catching where coordinating atoms were deprotonated
        ### WORKING -> Does not work great for nitrogen compounds.
        for l, atom in enumerate(ob.OBMolAtomIter(ligobmol)):
            total_val = (io_ptable.valence_electrons[atom.GetAtomicNum()] + atom.GetTotalValence())
            close = np.argmin(np.abs(np.array(io_ptable.filled_valence_electrons)-total_val))
            atom.SetFormalCharge(int(atom.GetFormalCharge()-(io_ptable.filled_valence_electrons[close]-total_val)))
        new_smiles = get_smiles_obmol(ligobmol,canonicalize=True)
        ligand_smiles.append(new_smiles)
        if calc_coord_atoms:
            new_coord_atom_list = map_coord_ats_smiles(new_smiles, ligobmol, coord_atom_list)
            coord_atom_lists.append(sorted(new_coord_atom_list))
        else:
            coord_atom_lists = []
    if not return_info:
        return ligand_smiles,coord_atom_lists
    else:
        info_dict = dict()
        if len(met_inds) > 0:
            info_dict['metal'] = io_ptable.elements[anums[met_inds[0]]]
            info_dict['metal_ind'] = met_inds[0]
        else:
            info_dict['metal'] = None
            info_dict['metal_ind'] = None
        lig_obmols = [get_obmol_smiles(smi,addHydrogens=True,
                    neutralize=False,
                    build=False) for smi in ligand_smiles]
        info_dict['lig_charges'] = [x.GetTotalCharge() for x in lig_obmols]
        lig_coord_ats = []
        if calc_coord_atoms:
            for i,lig_obmol in enumerate(lig_obmols):
                _,anums,_ = get_OBMol_coords_anums_graph(lig_obmol,get_types=False)
                lig_coord_ats.append(','.join([io_ptable.elements[x] for x in np.array(anums)[np.array(coord_atom_lists[i])]]))
            info_dict['lig_coord_ats'] = lig_coord_ats
        else:
            info_dict['lig_coord_ats'] = None
        return ligand_smiles,coord_atom_lists, info_dict


def get_canonical_label(obmol):
    """Create a canonical label for the molecule using pynauty.
    Warning - can fail/hang on particularly gnarly graphs.
    """
    _,anums,graph = get_OBMol_coords_anums_graph(obmol,get_types=True)
    grph = pnGraph(graph.shape[0])
    colors = get_vertex_coloring(anums) # Color by atom_type
    for n in range(graph.shape[0]):
        neighs = np.nonzero(np.ravel(graph[n]))[0]
        grph.connect_vertex(n, neighs.tolist())
    grph.set_vertex_coloring(colors)
    can_label = canon_label(grph) # Pynauty canonical label
    return can_label


def get_vertex_coloring(anums):
    """ Create set of like atoms by atomic number for graph labelling
    """
    if isinstance(anums,list): # Convert to array
        anums = np.array(anums)
    sym_labels = sorted(list(set(anums)))
    out_sets = []
    for anum in sym_labels:
        ind_set = set(np.where(anums == anum)[0])
        out_sets.append(ind_set)
    return out_sets


def map_coord_ats_smiles(lig_smiles, lig_obmol, coord_atoms):
    """map_coord_ats_smiles 
    Map the 3d structure of a ligand with encoded coordinating atom information
    to the smiles string of the ligand to get the correct smicat atoms
    Basic routine is done by converting both smiles and 3d structure to 
    molecular graphs, then using the coordinating atom information in addition 
    to the pynauty canonicalized form of the molecular graphs (colored by atom type)
    to map the 3d structure graph to the smiles graph.
    """
    tmp1 = get_obmol_smiles(lig_smiles)
    can1 = get_canonical_label(tmp1) 
    can2 = get_canonical_label(lig_obmol) # Know indices
    smicat = []
    for atom in coord_atoms: # Match canonicalized graph indices
        a_ind = can2.index(atom)
        smicat.append(can1[a_ind])
    return smicat

def get_fingerprint(obmol,fp='FP2'):
    """get_fingerprint 
    Gets the fingerprint for an obmol molecule.

    Parameters
    ----------
    obmol : ob.OBMol
        molecule
    fp : str, optional
        type of fingerprint, by default 'FP2'

    Returns
    -------
    fp : bit vector
        openbabel fingerprint of the molecule
    """
    pybelmol = pybel.Molecule(obmol)
    fp = pybelmol.calcfp(fp)
    return fp

# Main
if (__name__ == '__main__'):
    # Variables
    smiles = "NCCN"
    print(smiles2xyz(smiles))