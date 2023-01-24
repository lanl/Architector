"""
Implements a distance geometry conformer search routine biased to specific geometries
Note that DG is currently performed and hydrogens are added by OBmol after 
For cases where hydrogen is not a coordinating atom.

Influenced by distgeom.py script from molSimplify and molassembler.
Adapted by Michael Taylor

Adapted from:

[1] J. M. Blaney and J. S. Dixon, "Distance Geometry in Molecular Modeling", 
                                 in Reviews in Computational Chemistry, VCH (1994)

[2] G. Crippen and T. F. Havel, "Distance Geometry and Molecular Conformation", 
                                  in Chemometrics Research Studies Series, Wiley (1988)
With concepts from:
 
  [1] https://en.wikipedia.org/wiki/Kabsch_algorithm
"""

import numpy as np
import itertools
import scipy
import architector.io_ptable as io_ptable
import architector.io_molecule as io_molecule
import architector.io_obabel as io_obabel
import architector.arch_context_manage as arch_context_manage


from openbabel import openbabel
from scipy import optimize
from scipy.optimize import quadratic_assignment
from scipy.spatial.transform import Rotation as Rot

from ase import Atom
from ase.optimize.bfgslinesearch import BFGSLineSearch
import ase.constraints as ase_con
from xtb.ase.calculator import XTB

from numba import jit
import warnings

warnings.filterwarnings('ignore') # Supress numpy warnings.

def set_XTB_calc(ase_atoms):
    """set_XTB_calc 
    assign xtb calculator to atoms instance!

    Parameters
    ----------
    ase_atoms : ase.atoms.Atoms
        atoms to assign calculator to
    """
    ase_atoms.set_initial_charges(np.zeros(len(ase_atoms)))
    ase_atoms.set_initial_magnetic_moments(np.zeros(len(ase_atoms)))
    calc = XTB(method="GFN-FF") # Default to only GFN-FF for ligand conformer relaxation.
    #########################################################
    ########### Calculator Now Set! #########################
    #########################################################
    ase_atoms.calc = calc
    return ase_atoms

def calc_angle(a, b, c):
    """calc_angle
    Apply inverse cosine to find angle between a-b(vertex)-c
        
    Parameters
    ----------
        a : np.ndarray
            Coordinates of a
        b : np.ndarray
            Coordinates of b.
        c : list
            Coordinates of c.
        
    Returns
    -------
        theta : float
            a-b-c angle theta in degrees.
    """
    v1 = np.array(a)-np.array(b)
    v2 = np.array(c)-np.array(b)
    theta = np.degrees(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
    return theta

def symmetricize(arr1D):
    """symmetricize take in a 1d array and map it to 2D
    Primarily Used for determining 2D locations of atomic symbols
    in bonds

    Parameters
    ----------
    arr1D : np.ndarray
        1D array

    Returns
    -------
    np.ndarray
        2D array 
    """
    ID = np.arange(arr1D.size)
    return arr1D[np.abs(ID - ID[:,None])]

def get_bounds_matrix(allcoords, molgraph, natoms, catoms, shape, ml_dists, vdwradii, anums,
                      isCp=False, cp_catoms=[], bond_tol=0.1, angle_tol=0.1, ca_bond_tol=0.2,
                      ca_angle_tol=0.1,h_bond_tol=0,h_angle_tol=0,metal_center_lb_multiplier=1,
                      add_angle_constraints=True):
    """get_bounds_matrix
    Generate distance bounds matrices. The basic idea is outlined in ref [1].
    Bond constraints from mmff64-relaxed conformer used for organic section.
    M-L bond constraints from sum of covalent radii. (L=coordinating atom)
    L-L bond constraints from ideal angles
    All other atoms lower bounds specified by sum of vdwradii.
    For M-L-second-nearest-neighbor, the lower bounds are sum of vdwradii*1.2
    - to limit metal center crowding.

    Parameters
    ----------
        allcoords : np.ndarray
            NX3 matrix of coordinates of the atoms
        molgraph : np.ndarray
            molecular graph of the molecule
        natoms : int
            Number of atoms in the molecule (N)
        catoms : list
            List of ligand connection atoms.
        shape : np.ndarray
            matrix containing desired L-M-L angles
        ml_dists : np.ndarray
            N array containing desired M-L distances
        vdwradii : np.ndarray
            N array containing the vdwradii for each atom by index
        anums : list
            N array with lists of the atomic numbers of the ligand
        isCp : bool
            Whether the ligand is a Cp-style ligand
        cp_catoms : list
            list of Cp-ligand connecting atoms
        bond_tol : float, optional
            bond tolerance factor for defined bond distances for non-hydrogens. between 0 and 1, default 0.1
        angle_tol : float, optional
            1-3 "angle" distance tolerance for non-hydrogens, default 0.1
        h_bond_tol : float
            bond tolerance factor for defined bond distances forhydrogens, default 0
        h_angle_tol : float, optional
            1-3 "angle" distance tolerance for hydrogens, default 0
        metal_center_lb_multiplier : float, optional
            Lower bound multiplier for the metal center distances, default 1. 
        add_angle_constraints : bool, optional
            add angle constraints for Ca-M-Ca bonds or not, default True

    Returns
    -------
        LB : np.array
            Lower bound matrix
        UB : np.array
            Upper bound matrix
    """
    if isinstance(anums,list):
        anums = np.array(anums)
    if isinstance(vdwradii,list):
        vdwradii = np.array(vdwradii)
    LB = np.zeros((natoms, natoms))  # initialize lower bound
    UB = np.zeros((natoms, natoms))  # initialize upper bound, both symmetric
    # Set constraints for all bonding atoms excluding the dummy metal atom
    dummy_idx = natoms-1

    depth = scipy.sparse.csgraph.dijkstra(scipy.sparse.csgraph.csgraph_from_dense(molgraph))
    distmat = np.sqrt(np.sum((allcoords[:, np.newaxis, :] - allcoords[np.newaxis, :, :]) ** 2, axis = -1))

    next_neighs = []
    cpneighs = []
    if (not isCp):
        next_neighs = np.where(depth[dummy_idx] == 2)[0]
    else:
        tcp = [x for x in catoms if x not in cp_catoms] # Pull out non-cp ring coords
        for i,catom in enumerate(tcp):
            next_neigh = np.nonzero(np.ravel(molgraph[catom]))[0]
            new_neighs = [x for x in next_neigh if (x not in next_neighs) and (x not in catoms)]
            next_neighs += new_neighs
        for i,catom in enumerate(cp_catoms):
            next_neigh = np.nonzero(np.ravel(molgraph[catom]))[0]
            new_neighs = [x for x in next_neigh if (x not in cpneighs) and (x not in catoms)]
            cpneighs += new_neighs
        
    # 1-2 constraints: UB = LB = BL
    H_i1j2_inds = np.where((depth == 1) & symmetricize(anums == 1))
    LB[H_i1j2_inds] = distmat[H_i1j2_inds] * (1 - h_bond_tol)
    UB[H_i1j2_inds] = distmat[H_i1j2_inds] * (1 + h_bond_tol)
    i1j2_inds = np.where((depth == 1) & symmetricize(anums != 1))
    LB[i1j2_inds] = distmat[i1j2_inds] * (1 - bond_tol)
    UB[i1j2_inds] = distmat[i1j2_inds] * (1 + bond_tol)

    # 1-3 constraints: UB = LB = BL
    # "Angles" constraints
    H_i1k2_inds = np.where((depth == 2) & symmetricize(anums == 1))
    LB[H_i1k2_inds] = distmat[H_i1k2_inds] * (1 - h_angle_tol)
    UB[H_i1k2_inds] = distmat[H_i1k2_inds] * (1 + h_angle_tol)
    i1k2_inds = np.where((depth == 2) & symmetricize(anums != 1))
    LB[i1k2_inds] = distmat[i1k2_inds] * (1 - angle_tol)
    UB[i1k2_inds] = distmat[i1k2_inds] * (1 + angle_tol)

    # # 1-4 constraints: Possible to add - ususally just makes it a bit slower.
    # H_i1k2_inds = np.where((depth == 3) & symmetricize(anums == 1))
    # LB[H_i1k2_inds] = distmat[H_i1k2_inds] * (0.8)
    # UB[H_i1k2_inds] = distmat[H_i1k2_inds] * (1.2)
    # i1k2_inds = np.where((depth == 3) & symmetricize(anums != 1))
    # LB[i1k2_inds] = distmat[i1k2_inds]* (0.8)
    # UB[i1k2_inds] = distmat[i1k2_inds]* (1.2)

    vdw_summat = vdwradii + vdwradii.reshape(-1,1)

    # Other Depths
    H_i1z2_inds = np.where((depth > 2) & symmetricize(anums == 1))
    LB[H_i1z2_inds] = vdw_summat[H_i1z2_inds] * 1.2
    UB[H_i1z2_inds] = 100
    i1z2_inds = np.where((depth > 2) & symmetricize(anums != 1))
    LB[i1z2_inds] = vdw_summat[i1z2_inds]
    UB[i1z2_inds] = 100

    # Set constraints for atoms bonded to the dummy metal atom
    for i,catom in enumerate(catoms):
        # Set 1-2 constraints
        UB[catom,dummy_idx] = ml_dists[i] * (1 + ca_bond_tol)
        UB[dummy_idx,catom] = ml_dists[i] * (1 + ca_bond_tol)
        LB[catom,dummy_idx] = ml_dists[i] * (1 - ca_bond_tol)
        LB[dummy_idx,catom] = ml_dists[i] * (1 - ca_bond_tol)

    if (len(catoms) > 1) and (add_angle_constraints):
        # Set 1-3 contraints for ligating atoms -> Cosine rule
        for i in range(len(catoms[:-1])):
            for j in range(i+1, len(catoms)):
                angle = shape[i,j]
                theta = np.pi*angle/180
                lig_distance = np.sqrt(ml_dists[i]**2+ml_dists[j]**2-2*ml_dists[i]*ml_dists[j]*np.cos(theta))
                UB[catoms[i],catoms[j]] = lig_distance * (1 + ca_angle_tol)
                UB[catoms[j],catoms[i]] = lig_distance * (1 + ca_angle_tol)
                LB[catoms[i],catoms[j]] = lig_distance * (1 - ca_angle_tol)
                LB[catoms[j],catoms[i]] = lig_distance * (1 - ca_angle_tol)
    elif (len(catoms) > 1): # Set to broad range
        for i in range(len(catoms[:-1])):
            for j in range(i+1, len(catoms)):
                angle = shape[i,j]
                theta = np.pi*angle/180
                lig_distance = np.sqrt(ml_dists[i]**2+ml_dists[j]**2-2*ml_dists[i]*ml_dists[j]*np.cos(theta))
                UB[catoms[i],catoms[j]] = vdwradii[-1] * 3
                UB[catoms[j],catoms[i]] = vdwradii[-1] * 3
                LB[catoms[i],catoms[j]] = vdw_summat[catoms[i],catoms[j]] * 0.7
                LB[catoms[j],catoms[i]] = vdw_summat[catoms[i],catoms[j]] * 0.7

    # Adding depth (graph distance) from the metal constraints!
    m_depth = depth[-1]

    for i in range(natoms-1): # Iterate through and assign M-atom bounds
        j = dummy_idx
        if (i in cpneighs): # Don't force nearest neighbors quite so far away for cp ligands
            LB[i,j] = (vdwradii[i] + vdwradii[j])*0.9*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*0.9*metal_center_lb_multiplier
            UB[i,j] = 10
            UB[j,i] = 10
        elif (i in next_neighs) and (anums[i] != 1): # Make next nearest neighbors longer than vdwrad sum.
            LB[i,j] = (vdwradii[i] + vdwradii[j])*1.2*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*1.2*metal_center_lb_multiplier
            UB[i,j] = 20
            UB[j,i] = 20
        elif (i in next_neighs) and (anums[i] == 1): # Make next nearest hydrogen neighbors a little closer
            LB[i,j] = (vdwradii[i] + vdwradii[j])*1.1*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*1.1*metal_center_lb_multiplier
            UB[i,j] = 20
            UB[j,i] = 20
        elif (m_depth[i] < 3):
            continue
        elif (m_depth[i] < 4) and (anums[i] != 1): # Encourage further away conformers.
            LB[i,j] = (vdwradii[i] + vdwradii[j])*1.3*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*1.3*metal_center_lb_multiplier
            UB[i,j] = 50
            UB[j,i] = 50
        elif (m_depth[i] < 4) and (anums[i] == 1): # Encourage further away conformers.
            LB[i,j] = (vdwradii[i] + vdwradii[j])*1.1*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*1.1*metal_center_lb_multiplier
            UB[i,j] = 50
            UB[j,i] = 50
        # Allow closer atoms for huge lanthanides
        elif (m_depth[i] >= 4) and (anums[j] > 56):
            LB[i,j] = (vdwradii[i] + vdwradii[j])*1.2*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*1.2*metal_center_lb_multiplier
            UB[i,j] = 100
            UB[j,i] = 100 # Set upper bound very high
        # Force non-hydrogens further away at greater graph depths
        elif (m_depth[i] >= 4) and (anums[i] != 1):
            LB[i,j] = (vdwradii[i] + vdwradii[j])*1.5*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*1.5*metal_center_lb_multiplier
            UB[i,j] = 100
            UB[j,i] = 100 # Set upper bound very high
        # Force hydrogens not so far away for greater graph depths
        elif (m_depth[i] >= 4) and (anums[i] == 1):
            LB[i,j] = (vdwradii[i] + vdwradii[j])*1.3*metal_center_lb_multiplier
            LB[j,i] = (vdwradii[i] + vdwradii[j])*1.3*metal_center_lb_multiplier
            UB[i,j] = 100
            UB[j,i] = 100 # Set upper bound very high

    return LB, UB

@jit(nopython=True)
def triangle(LB, UB, natoms):
    """triangle
    Triangle inequality bounds smoothing. From ref [2], pp. 252-253.
    Scales poorly - using jit compilation for faster evalulation. 

    Parameters
    ----------
        LB : np.array
            Lower bounds matrix.
        UB : np.array
            Upper bounds matrix.
        natoms : int
            Number of atoms in the molecule.
        
    Returns
    -------
        LL : np.array
            Lower triangularized bound matrix
        UL : np.array
            Upper triangularized bound matrix      
    """
    LL = LB
    UL = UB
    for k in range(natoms):
        for i in range(natoms-1):
            for j in range(i+1,natoms):
                if UL[i,j] > UL[i,k] + UL[k,j]:
                    UL[i,j] = UL[i,k] + UL[k,j]
                    UL[j,i] = UL[i,k] + UL[k,j]
                if LL[i,j] < LL[i,k] - UL[k,j]:
                    LL[i,j] = LL[i,k] - UL[k,j]
                    LL[j,i] = LL[i,k] - UL[k,j]
                else:
                    if LL[i,j] < LL[j,k] - UL[k,i]:
                        LL[i,j] = LL[j,k] - UL[k,i]
                        LL[j,i] = LL[j,k] - UL[k,i]
                    if LL[i,j] > UL[i,j]:
                        raise ValueError('Bounds Incorrect.')
    return LL, UL

def metrize(LB, UB, natoms, non_triangle=False, debug=False):
    """metrize
    Metrization selects random in-range distances. Copied from ref [2], pp. 253-254.
    Scales O(N^3). 

    Parameters
    ----------
        LB : np.array
            Lower bounds matrix.
        UB : np.array
            Upper bounds matrix.
        natoms : int
            Number of atoms in the molecule.
        
    Returns
    -------
        D : np.array
            Distance matrix.  
    """
    D = np.zeros((natoms, natoms))
    LB, UB = triangle(LB, UB, natoms)
    for i in range(natoms-1):
        for j in range(i+1, natoms):
            if UB[i,j] < LB[i,j]:  # ensure that the upper bound is larger than the lower bound
                UB[i,j] = LB[i,j]
            # LB,UB = triangle(LB,UB,natoms) # Uncomment for full metrization 
            D[i,j] = np.random.uniform(LB[i,j], UB[i,j])
            D[j,i] = D[i,j]
            LB[i,j] = D[i,j]
            UB[i,j] = D[i,j]
            LB[j,i] = D[i,j]
            UB[j,i] = D[i,j]
    
    if non_triangle:
        #For pairs involving the metal, set the distance to 100 Angstroms regardless of the triangle rule
        #This encourages the algorithm to select conformations that don't crowd the metal
        for j in range(natoms):
            if UB[natoms-1,j] < LB[natoms-1,j]:  # ensure that the upper bound is larger than the lower bound
                UB[natoms-1,j] = LB[natoms-1,j]
            D[natoms-1][j] = 100
            D[j][natoms-1] = D[natoms-1][j]
    return D

@jit(nopython=True)
def get_cm_dists(D, natoms):
    """get_cm_dists
    Get distances of each atom to center of mass given the distance matrix. 
    Copied from ref [2], pp. 309.

    Parameters
    ----------
        D : np.array
            Distance matrix.
        natoms : int
            Number of atoms in the molecule.
        
    Returns
    -------
        D0 : np.array
            Vector of distances from center of mass.
        status : bool
            Flag for successful search.
    """
    D0 = np.zeros(natoms)
    for i in range(natoms):
        for j in range(natoms):
            D0[i] += D[i,j]**2/natoms
        for j in range(natoms):
            for k in range(j, natoms):
                D0[i] -= (D[j,k])**2/natoms**2
        D0[i] = np.sqrt(D0[i])
    return D0

@jit(nopython=True)
def get_metric_matrix(D, D0, natoms):
    """get_metric_matrix
    Get metric matrix from distance matrix and cm distances 
    Copied from ref [2], pp. 306.

    Parameters
    ----------
        D : np.ndarray
            Distance matrix.
        D0 : np.ndarray
            Vector of distances from center of mass.
        natoms : int
            Number of atoms in the molecule.
        
    Returns
    -------
        G : np.ndarray
            Metric matrix
    """
    G = np.zeros((natoms, natoms))
    for i in range(natoms):
        for j in range(natoms):
            G[i,j] = (D0[i]**2 + D0[j]**2 - D[i,j]**2)/2
    return G

def get_3_eigs(G, natoms):
    """get_3_eigs
    Gets 3 largest eigenvalues and corresponding eigenvectors of metric matrix
        
    Parameters
    ----------
        G : np.ndarray
            Metric matrix.
        natoms : int
            Number of atoms in the molecule.
        
    Returns
    -------
        L : np.ndarray
            Three largest eigenvalues
        V : np.ndarray
            Eigenvectors corresponding to largest eigenvalues.
    """
    L = np.zeros((3, 3))
    V = np.zeros((natoms, 3))
    l, v = np.linalg.eigh(G)
    for i in [0, 1, 2]:
        L[i,i] = np.sqrt(np.max(l[natoms-1-i], 0))
        V[:, i] = v[:, natoms-1-i]
    return L, V

@jit(nopython=True)
def distance_error(x, *args):
    """distance_error
    Computes distance error function for scipy optimization. 
    Copied from E3 in pp. 311 of ref. [1]
        
    Parameters
    ----------
        x : np.ndarray
            1D array of coordinates to be optimized.
        *args : dict
            Other parameters for optimization (needed for scipy.optimize)
        
    Returns
    -------
        E : np.ndarray
            Objective function
    """
    E = 0.0
    LB, UB, natoms = args
    for i in range(natoms-1):
        for j in range(i+1, natoms):
            ri = np.array([x[3*i], x[3*i+1], x[3*i+2]])
            rj = np.array([x[3*j], x[3*j+1], x[3*j+2]])
            dij = float(np.linalg.norm(ri-rj))
            uij = float(UB[i][j])
            lij = float(LB[i][j])
            E += (dij**2/(uij**2) - 1)**2
            E += (2*lij**2/(lij**2 + dij**2) - 1)**2
    return E

@jit(nopython=True)
def dist_error_gradient(x, *args):
    """dist_error_gradient
    Computes gradient of distance error function for scipy optimization.
    Copied from E3 in pp. 311 of ref. [1]
        
    Parameters
    ----------
        x : np.array
            1D array of coordinates to be optimized.
        *args : dict
            Other parameters (refer to scipy.optimize docs)
        
    Returns
    -------
        g : np.array
            Objective function gradient
    """
    LB, UB, natoms = args
    g = np.zeros(3*natoms)
    for i in range(natoms):
        jr = list(range(natoms))
        jr.remove(i)
        for j in jr:
            ri = np.array([x[3*i], x[3*i+1], x[3*i+2]])
            rj = np.array([x[3*j], x[3*j+1], x[3*j+2]])
            dij = np.linalg.norm(ri-rj)
            uij = UB[i][j]
            lij = LB[i][j]
            g[3*i] += (4*((dij/uij)**2-1)/(uij**2) - (8/lij**2)*(2*(lij**2 /
                                (lij**2+dij**2))-1)/((1+(dij/lij)**2)**2))*(x[3*i]-x[3*j])  # xi
            g[3*i+1] += (4*((dij/uij)**2-1)/(uij**2) - (8/lij**2)*(2*(lij**2 /
                                (lij**2+dij**2))-1)/((1+(dij/lij)**2)**2))*(x[3*i+1]-x[3*j+1])  # yi
            g[3*i+2] += (4*((dij/uij)**2-1)/(uij**2) - (8/lij**2)*(2*(lij**2 /
                                (lij**2+dij**2))-1)/((1+(dij/lij)**2)**2))*(x[3*i+2]-x[3*j+2])  # zi
    return g

def get_ideal_angles(ligating_coords, metal_coords=np.array((0,0,0))):
    """get_ideal_angles
    Determines the relative angular positioning of ligating atoms

    Parameters
    ----------
    ligating_coords : list
        list of ligating coordinates
    metal_coords : np.ndarray, optional
        location of the metal - default (0,0,0)

    Returns
    -------
    angles_out : np.ndarray
        angles matrix between each pair of coordinating atoms through the metal
    """
    metal_coords = metal_coords
    angles_out = np.zeros((len(ligating_coords), len(ligating_coords)))
    if len(ligating_coords) > 1:
        inds = list(range(len(ligating_coords)))
        for (i,j) in itertools.combinations(inds,2):
            val = calc_angle(ligating_coords[i],metal_coords, ligating_coords[j])
            angles_out[i,j] = val
            angles_out[j,i] = val
    angles_out[0,0] = 0
    return angles_out

def detect_cps(OBMol, ligcoordList):
    """detect_cps 
    Take in ligcoordList and OBMol of smiles -> detect if Cp binding requested

    Parameters
    ----------
    OBMol : OBMol
        OBmol of molecule (without metal added)
    ligcoordList : list
        list of lists containing coord atom information

    Returns
    -------
    isCp : bool
        if the structure passed is requesting Cp generation
    cp_rings : list
        list of lists of indices of cp ring coordinating atoms
    shared_coords : list 
        list of lists of the corecoordList indices shared by each ring
    """
    # Are there sets of coordinating atoms that are given in inputs? - based on if list passed
    multiflag_lig_con_sets = np.array([','.join([str(y) for y in sorted(x[1])]) for x in ligcoordList if isinstance(x[1],(list,np.ndarray))])
    isCp = False
    cp_rings = []
    shared_coords = []
    if len(multiflag_lig_con_sets) > 0:
        isCp = True
        shared_coord_indices = np.unique(multiflag_lig_con_sets)
        multiflag_lig_coords = np.array([x[0] for x in ligcoordList if isinstance(x[1],list)])
        cp_rings = []
        if len(shared_coord_indices) > 1: # This is for multiple-CP bound ligands: see UTUGUN CSD refcode for example
            # Pull out the coordination indices that are mapped together in rings
            for shared in shared_coord_indices:
                tcord_list = []
                for i,x in enumerate(multiflag_lig_con_sets):
                    if x == shared:
                        tcord_list.append(int(multiflag_lig_coords[i]))
                rings = OBMol.GetSSSR()
                for ring in rings:
                    if all(ring.IsInRing(x+1) for x in tcord_list):
                        cp_rings.append(tcord_list)
                        shared_coords.append([int(x) for x in shared.split(',')])
        else:
            # Flag sets of coordination indices
            rings = OBMol.GetSSSR()
            for ring in rings:
                if all([ring.IsInRing(int(x)+1) for x in multiflag_lig_coords]):
                    cp_rings.append(multiflag_lig_coords)
                    shared_coords.append([int(x) for x in shared_coord_indices[0].split(',')])
    return isCp, cp_rings, shared_coords

def find_best_plane_n_points(points):
    """find_best_plane_n_points 
    Calculate the plane of best fit for n points.

    Parameters
    ----------
    points : np.ndarray
        array of points to calculate plane of best fit.

    Returns
    -------
    centroid : np.ndarray
        length 3 array of the centroid of the given points
    normal : np.ndarray
        normal vector to the plane of best fit.
    """
    centroid = points.mean(axis=0)
    xyzR = points - centroid                      
    _, _, v= np.linalg.svd(xyzR)
    normal= v[2]                                 
    normal= normal / np.linalg.norm(normal)
    return centroid,normal

def manage_cps(OBmol, cp_rings, shared_coords, ligcoordList, corecoordList,
               metal, covrad_metal=None):
    """manage_cps
    bin Cp rings into grouped coord coordination vectors based on input
    then split into geometrically-assigned vectors

    Parameters
    ----------
    OBmol : OBmol
        openbabel of the non-metal-added MMFF-relaxed ligand.
    cp_rings : list
        list of the indices of each cP ring
    shared_coords : list
        list of lists of the corecoordList indices shared by each ring
    ligcoordList : list
        list of list of given indices and corecoordList index for binding site
    corecoordList : list
        list of lists of the core binding geometry locations
    metal : str
        identity of the metal -> for calculating how far away to initialize the cP rings
    covrad_metal : float
        covalent radii of the metal, default None

    Returns 
    ----------
    outligcoordList : list
        recreated list of list of coordination sites and ideal locations accounting for Cp
    outcorecoordList : list
        out core coordination list matching outligcoordList
    cp_catoms : list
        list of catoms that are in cp lignads
    catoms : list
        new sorted list of all the catoms
    """
    allcoords, anums, _ = io_obabel.get_OBMol_coords_anums_graph(OBmol)
    metal_elem_ind = io_ptable.elements.index(metal)
    if isinstance(covrad_metal,float):
        met_rcov1 = covrad_metal
    else:
        met_rcov1 = io_ptable.rcov1[metal_elem_ind]
    anums = np.array(anums)
    core_coords = list(range(len(corecoordList))) # List for keeping track of which (if any) original core
    # Coordination vectors maintined after aligning cp ligand -> preserve info
    outcorecoordList = []
    n_ind = 0 # keep track of how many points assigned
    outligcoordList = []

    for i,shared in enumerate(shared_coords):
        cp_ring_inds = np.array(cp_rings[i]) # Get indices of ring atoms

        elems_cp_bound = anums[cp_ring_inds] # Pull out the element identities of the cp-bound ring
        unique,counts = np.unique(elems_cp_bound,return_counts=True)
        most_common_elem_covrad = io_ptable.rcov1[unique[np.argmax(counts)]]
        scaledist = 1*(most_common_elem_covrad + met_rcov1) # Rescale default distance

        newbasis_vect = np.array((0.0,0.0,0.0))
        # Generate centroid
        for s in shared:
            newbasis_vect += np.array(corecoordList[s])
            if s in core_coords: # Remove indice as "used" for cp ligand
                core_coords.remove(s)
        newbasis_vect = newbasis_vect/np.linalg.norm(newbasis_vect)*scaledist # Push back out
        cp_ring_coords = allcoords[cp_ring_inds]
        centroid, normal = find_best_plane_n_points(cp_ring_coords) # Rotate to X axis
        cp_ring_coords = cp_ring_coords - centroid
        r = Rot.align_vectors(np.array((1.,0.,0.)).reshape(1,-1), normal.reshape(1,-1))
        rot_cp_ring_coords = r[0].apply(cp_ring_coords) # Rotate to x axis
        rot_cp_ring_coords[:,0] = rot_cp_ring_coords[:,0] + scaledist # Shift out to new distance

        # Rotate centroid to the new basis position
        r1 = Rot.align_vectors(newbasis_vect.reshape(1,-1),np.array((scaledist,0.,0.)).reshape(1,-1)) 
        new_core_coords = r1[0].apply(rot_cp_ring_coords) # Rotate all coordinates to area around new basis position
        for j,coord in enumerate(new_core_coords): # Set the cp ring coords to correct inds
            outcorecoordList.append(coord.tolist())
            outligcoordList.append([int(cp_ring_inds[j]),n_ind])
            n_ind += 1

    cp_catoms = [x[0] for x in outligcoordList]
    catoms = [x[0] for x in outligcoordList]

    for core_coord in core_coords: # Map remaining coord sites to correct positions
        if any([True for x in ligcoordList if (x[1] == core_coord)]): # Check if true first
            outcorecoordList.append(corecoordList[core_coord])
            tligref = [x for x in ligcoordList if (x[1] == core_coord)][0]
            tligref[1] = n_ind
            outligcoordList.append(tligref)
            n_ind += 1
            catoms += [tligref[1]]
    return outligcoordList, outcorecoordList, cp_catoms, catoms

def reorder_ligcoordList(coords, catoms, shape, ligcoordList, isCp=False):
    """reorder_ligcoordList attempts to reorder the ligcoordList/catom
    indices to match closest to the ideal shape from UFF relaxation.
    Uses the quadratic assignment (travelling salesman) algorithm
    implemented in scipy.optimize

    Parameters
    ----------
    coords : np.ndarray
        coordinates of the ligand/metal from UFF relaxation without angle constraints.
    catoms : list
        coordinating atoms
    shape : np.ndarray
        array of the ideal angles to fit to.
    ligcoordList : list
        list of list of ligand coordinating atoms
    isCp : bool, optional
        whether this is a cp ligand, by default False

    Returns
    -------
    new_ligcoordList : list
        reordered list of ligand coordinating atoms (needed for rotation assignment)
    new_catoms : list
        new coordinating atoms
    """
    if (len(catoms) > 2) and (not isCp): # Only do for multidentates where ordering may be wrong.
        actmat = get_ideal_angles(coords[catoms],metal_coords=coords[-1])
        out = quadratic_assignment(shape,actmat,options={'maximize':True})
        outorder = out['col_ind']
        new_catoms = np.array(catoms)[outorder]
        new_catoms = new_catoms.tolist()
        new_ligcoordList = []
        for i,val in enumerate(ligcoordList):
            new_ligcoordList.append([new_catoms[i],val[1]])
    else:
        new_ligcoordList = ligcoordList.copy()
        new_catoms = catoms
        pass
    return new_ligcoordList, new_catoms

def clean_conformation_ff(X, OBMol, catoms, shape, graph, 
                          atomic_numbers, ligcoordList, add_angle_constraints=True,
                          isCp=False, cp_catoms=[], skip_mff=False,add_hydrogens=True,
                          debug=False
                          ):
    """clean_conformation_ff
    Further cleans up with OB FF and saves to a new ase object.
        
    Parameters
    ----------
    X : np.array
        Array of distgeom generated coordinates.
    catoms : list
        List of coordinating atoms used to generate FF constraints.
    OBMol : OBmol
        OBmol of original molecule with dummy metal added
    shape : np.ndarray
        Matrix containing ideal angles for coordinating atoms through metal
    graph : np.ndarray
        NXN array representing the molecular graph
    atomic_numbers : np.ndarray
        N array containing the atomic numbers -> used for flagging hydrogens as 
        second-nearest neighbors to the metal.
    add_angle_constraints : bool
        Whether or not to add angle constraints to force ligand into position, default False
    isCp : bool, optional
        whether this is a cp ligand, by default False
    skip_mff : bool, optional
        wheter to skip the mmff94 intermediate relaxatino, default False
    add_hydrogens : bool, optional
        Whether hydrogens need to be added to the structure, default True
    debug : bool, optional
        Whether to print debug messages.
        
    Returns
    -------
    ase_atoms : ase.Atoms
        ff-relaxed conformer in ase atoms
    fail : bool
        if any of the FF steps failed to converge.
    final_relax : bool
        whether or not to relax at the final step.
    tligcoordList : list
        re-ordered ligcoord list based on the actual geometry from ff relaxation
    """
    ##### STEP 1 Relaxation with UFF ######
    last_atom_index = OBMol.NumAtoms()
    fail = False
    # Set metal to zero
    metal_coords = (X[last_atom_index-1,0],X[last_atom_index-1,1],X[last_atom_index-1,2])
    # set coordinates using OBMol to keep bonding info
    for i, atom in enumerate(openbabel.OBMolAtomIter(OBMol)):
        atom.SetVector(X[i, 0]-metal_coords[0], X[i, 1]-metal_coords[1], X[i, 2]-metal_coords[2])
    if (not isCp): # Don't use UFF on cp ligands -> really does not work!
        #First stage of cleaning takes place with the metal still present
        constr = openbabel.OBFFConstraints()
        ff = openbabel.OBForceField.FindForceField('UFF')
        constr.AddAtomConstraint(int(last_atom_index))
        s = ff.Setup(OBMol,constr)
        if not s:
            if debug:
                print('UFF setup failed')
            fail=True
        try:
            for i in range(200):
                ff.SteepestDescent(10)
                ff.ConjugateGradients(10)
            ff.GetCoordinates(OBMol)
            if add_hydrogens: # Add hydrogens back only after initial optimization
                OBMol.AddHydrogens()
                constr = openbabel.OBFFConstraints()
                ff = openbabel.OBForceField.FindForceField('UFF')
                constr.AddAtomConstraint(int(last_atom_index))
                s = ff.Setup(OBMol,constr)
                for i in range(100):
                    ff.SteepestDescent(10)
                    ff.ConjugateGradients(10)
                ff.GetCoordinates(OBMol)
            coords, _, _ = io_obabel.get_OBMol_coords_anums_graph(OBMol, return_coords=True)
            # Reorder the coordinating atom assignment to more closely match unconstrained
            # UFF relaxation to the desired geometry.
            tligcoordList, catoms = reorder_ligcoordList(coords, catoms, shape, ligcoordList, isCp=isCp)
            if add_angle_constraints:
                if debug:
                    print('Finished initial UFF relaxation without angle constraints.')
                ff = openbabel.OBForceField.FindForceField('UFF')
                constr = openbabel.OBFFConstraints()
                constr.AddAtomConstraint(int(last_atom_index))
                # Add Ca-M-Ca angle constraints to coordinating atoms 
                # Helps encourage desired geometry
                for i in range(shape.shape[0]-1):
                    for j in range(i+1,shape.shape[0]):
                        constr.AddAngleConstraint(int(catoms[i]+1),int(last_atom_index),int(catoms[j]+1),float(shape[i,j]))
                s = ff.Setup(OBMol,constr)
                for i in range(200):
                    ff.SteepestDescent(10)
                    ff.ConjugateGradients(10)
                ff.GetCoordinates(OBMol)
        except:
            fail = True
            tligcoordList = ligcoordList.copy()
            if add_hydrogens:
                OBMol.AddHydrogens()
            if debug:
                print('Never Finished UFF relaxation.')
    else:
        tligcoordList = ligcoordList.copy()
        if add_hydrogens: # Re-add hydrogens after generation and UFF relaxation.
            OBMol.AddHydrogens()
    ########## Second stage of cleaning removes the metal - MMFF94 ################# 
    #### Uses constraints on the bonding atoms to ensure the binding conformation is maintained
    #### If not Cp ligand - otherwise - fully relaxing!

    #Delete the dummy metal atom that we added earlier for MMFF94
    metal_atom = OBMol.GetAtom(last_atom_index)
    OBMol.DeleteAtom(metal_atom)

    _, atomic_numbers, graph = io_obabel.get_OBMol_coords_anums_graph(OBMol, return_coords=False)

    mmff94_ok = io_obabel.check_mmff_okay(OBMol) # Check for match between chemistry and MMFF94

    # Set up force field
    if mmff94_ok:
        ff = openbabel.OBForceField.FindForceField("mmff94")
    else:
        ff = openbabel.OBForceField.FindForceField('UFF')

    if (not isCp):
        constr = openbabel.OBFFConstraints()
        for ca in catoms:
            constr.AddAtomConstraint(int(ca+1)) # Freeze coordinating atoms
            neighs = np.nonzero(graph[ca])[0]
            for n in neighs:
                # Freeze Hs on coordinating atoms (w/o metal will go into metal position)
                if atomic_numbers[n] == 1:
                    constr.AddAtomConstraint(int(n+1)) 
        s = ff.Setup(OBMol,constr)
    else:
        # Fixing tail/non-coordinating positions with MMFF -> attempt to fix hydrogen positions
        constr = openbabel.OBFFConstraints()
        for ca in catoms:
            constr.AddAtomConstraint(int(ca+1)) # Freeze coordinating atoms
        s = ff.Setup(OBMol,constr)

    if not s:
        if debug:
            print('MMFF setup failed')
        fail=True

    if not skip_mff:
        for i in range(200):
            ff.SteepestDescent(10)
            ff.ConjugateGradients(10)
        ff.GetCoordinates(OBMol)

    # Convert MMFF94 relaxed structure to ASE atoms
    ase_atoms_tmp = io_obabel.convert_obmol_ase(OBMol)
    final_relax = True

    #### Third stage of cleaning for only Cp - geometries ####
    if isCp: # GFN-FF needed for capturing Cp geometries
        constraintList = []
        mat = Atom(atomic_numbers[-1],(0,0,0))
        ase_atoms_tmp.append(mat) # Fix metal atom only for single cp ring
        fixCore = ase_con.FixAtoms(indices=[last_atom_index-1])

        if (len(cp_catoms) == 4) or (len(cp_catoms) == 3): #For small rings encourage staying in the same place!
            fixCon = ase_con.FixAtoms(indices=cp_catoms)
            constraintList.append(fixCon)
            final_relax = False
                
        constraintList.append(fixCore)
        ase_atoms_tmp.set_constraint(constraintList)
        ase_atoms_tmp = set_XTB_calc(ase_atoms_tmp)
        try: # Optimize structure with GFNFF
            with arch_context_manage.make_temp_directory() as _:
                dyn = BFGSLineSearch(ase_atoms_tmp, master=True)
                dyn.run(fmax=0.1, steps=1000)
        except Exception as e:
            if debug:
                print(e)
            fail = True
        ase_atoms_tmp.set_constraint() # Clear constraints
        ase_atoms_tmp.calc = None # Remove calculator
        ase_atoms_tmp.pop() # re-remove metal atom

    return ase_atoms_tmp, fail, final_relax, tligcoordList


def nonclean_conformation_ff(X, OBMol, catoms, shape, graph, 
                          atomic_numbers, ligcoordList, add_angle_constraints=True,
                          isCp=False, cp_catoms=[], skip_mff=False, add_hydrogens=False
                          ):
    """nonclean_conformation_ff -> Mostly used for debug/DG optimization
    Skips all OB FF and saves to a new ase object.

    Generaly not as useful except for debugging. Leaving here for reference.
        
    Parameters
    ----------
    X : np.array
        Array of distgeom generated coordinates.
    catoms : list
        List of coordinating atoms used to generate FF constraints.
    OBMol : OBmol
        OBmol of original molecule with dummy metal added
    shape : np.ndarray
        Matrix containing ideal angles for coordinating atoms through metal
    graph : np.ndarray
        NXN array representing the molecular graph
    atomic_numbers : np.ndarray
        N array containing the atomic numbers -> used for flagging hydrogens as 
        second-nearest neighbors to the metal.
    add_angle_constraints : bool
        Whether or not to add angle constraints to force ligand into position, default False
    isCp : bool, optional
        whether this is a cp ligand, by default False
    skip_mff : bool, optional
        wheter to skip the mmff94 intermediate relaxatino, default False
    add_hydrogens : bool, optional
        Whether hydrogens need to be added to the structure, default True
        
    Returns
    -------
    ase_atoms : ase.Atoms
        ff-relaxed conformer in ase atoms
    fail : bool
        if any of the FF steps failed to converge.
    final_relax : bool
        whether or not to relax at the final step.
    tligcoordList : list
        re-ordered ligcoord list based on the actual geometry from ff relaxation
    """
    # print('Starting FF relaxation.')
    last_atom_index = OBMol.NumAtoms()
    fail = False
    # Set metal to zero
    metal_coords = (X[last_atom_index-1,0],X[last_atom_index-1,1],X[last_atom_index-1,2])
    last_atom_index = OBMol.NumAtoms()
    # set coordinates using OBMol to keep bonding info
    for i, atom in enumerate(openbabel.OBMolAtomIter(OBMol)):
        atom.SetVector(X[i, 0]-metal_coords[0], X[i, 1]-metal_coords[1], X[i, 2]-metal_coords[2])
    coords, _, _ = io_obabel.get_OBMol_coords_anums_graph(OBMol, return_coords=True)
    # Reorder the connecting atom assignment to more closely match the actual geometry
    tligcoordList, catoms = reorder_ligcoordList(coords, catoms, shape, ligcoordList, isCp=isCp)
    #Delete the dummy metal atom that we added earlier
    metal_atom = OBMol.GetAtom(last_atom_index)
    OBMol.DeleteAtom(metal_atom)
    ase_atoms_tmp = io_obabel.convert_obmol_ase(OBMol,add_hydrogens=add_hydrogens,relax=False)
    fail = False
    final_relax = True
    return ase_atoms_tmp, fail, final_relax, tligcoordList


def set_position_align(ase_atoms, ligcoordList, corecoordList, isCp=False, debug=False,
                       rot_coord_vect=False, rot_angle=0):
    """set_position_align 
    Aligns the positions of the given coordinating atoms with the correct positions

    Parameters
    ----------
    ase_atoms : ase.Atoms
        Optimized ligand result in ase_atoms
    ligcoordList : list
        list of lists containing connecting atom info and desired coordination location
    corecoordList : list
        list of lists definining coordining environment (octahedral, tetrahedral....)
    isCp : bool, optional
        whether this is a cp ligand, by default False
    debug : bool, optional
        Print out additional information if different symmetries assigned
    rot_coord_vect : bool, optional

    Returns
    -------
    rotatedConformer : ase.Atoms
        ase atoms of rotated conformer
    minval : float
        angular loss on orientation (<1 means almost perfectly aligned)
    """
    # Make copy of atoms 
    conformerMolCpy = ase_atoms.copy()
    init_posits = conformerMolCpy.get_positions()
    fail = np.any(np.isnan(init_posits)) # Check for any nan in positions.
    if any([True for val in ligcoordList if isinstance(val[1],list)]) and (not fail):
        ### NOTE CURRENTLY ASSUMES ONLY ONE CP LIGAND in Multi-orientation styling
        shared = [val[1] for val in ligcoordList if isinstance(val[1],list)][0]
        cp_ring_inds = np.array([val[0] for val in ligcoordList if isinstance(val[1],list)])
        newbasis_vect = np.array((0.0,0.0,0.0))
        # Generate centroid
        for s in shared:
            newbasis_vect += np.array(corecoordList[s])
        cp_ring_coords = init_posits[cp_ring_inds]
        centroid, normal = find_best_plane_n_points(cp_ring_coords) # Rotate to X axis
        scaledist = np.linalg.norm(centroid)
        newbasis_vect = newbasis_vect/np.linalg.norm(newbasis_vect)*scaledist # Push back out
        cp_ring_coords = init_posits - centroid
        r = Rot.align_vectors(np.array((1.,0.,0.)).reshape(1,-1), normal.reshape(1,-1))
        rot_cp_ring_coords = r[0].apply(cp_ring_coords) # Rotate to x axis
        rot_cp_ring_coords[:,0] = rot_cp_ring_coords[:,0] + scaledist # Shift out to new distance
        r1 = Rot.align_vectors(newbasis_vect.reshape(1,-1),np.array((scaledist,0.,0.)).reshape(1,-1)) 
        newposits = r1[0].apply(rot_cp_ring_coords) # Rotate all coordinates to area around new basis position
        conformerMolCpy.set_positions(newposits)
        rotatedConformer = conformerMolCpy
        temp_mol = io_molecule.convert_io_molecule(rotatedConformer)
        temp_mol.dist_sanity_checks()
        sane = temp_mol.dists_sane
        minval = 1 # Set to 1 to not bias any conformers.
    elif (not fail):
        con_inds = np.array([val[1] for val in ligcoordList]) # Coordination indices
        ideal = np.array([np.asarray(corecoordList[val[1]]) for val in ligcoordList]) # Ideal positions to fit to
        actual = np.array([np.asarray(init_posits[val[0]]) for val in ligcoordList])
        if len(ideal) > 2: 
            # Add reflection planes in case of mirrored molecule generated
            mirrors = [[1,1,1],[-1,1,1]] # Add mirror vectors
            actual_mirrors = []
            for mirror in mirrors:
                actual_mirror = actual.copy()
                for i,val in enumerate(mirror):
                    actual_mirror[:,i] = val*actual_mirror[:,i]
                actual_mirrors.append(actual_mirror)
            r = []
            minval = 1000
            minind = 0
            for i,act in enumerate(actual_mirrors): # Check mirror images for matching rotations
                outr = Rot.align_vectors(ideal,act)
                if outr[1] < minval:
                    minind = i
                    minval = outr[1]
                    r = outr
            if (minval > 1.0) or isCp: # Take combinations of coordinating atoms to try and maximize overlap with coordination sites.
                inds = np.arange(0,len(ideal),1)
                orderings = np.array([np.array(x) for x in itertools.permutations(inds,len(inds))])
                if len(orderings) > 100:
                    ordering_inds = np.random.choice(np.arange(0,len(orderings),1),100)
                    orderings = orderings[ordering_inds]
                r = []
                minval = 1000
                save_ordering = []
                for ordering in orderings:
                    tactual = actual[ordering]
                    outr = Rot.align_vectors(ideal,tactual)
                    if outr[1] < minval:
                        r = outr
                        save_ordering = ordering
                        minval = r[1]
                if (not isCp) and (debug):
                    print('Warning: Your suggested connection sites and ligand cannot form an approximate guess.')
                    print('For this ligand/coord sites a possible suggested ordering for the selected connecting indices is: ')
                    print(con_inds[save_ordering])
                    print('I have defaulted to this order!!! Which may !!!! have shifted your desired outcome.')
                    print('But ensures the closest possible match between the desired coordination points and the molecule!')
            else:
                for i,val in enumerate(mirrors[minind]):
                    init_posits[:,i] = val*init_posits[:,i]
        elif len(ideal == 2): # No possible different symmetries here
            ideal = ideal
            actual = actual
            r = Rot.align_vectors(ideal,actual)
            minval = r[1]
        elif len(ideal == 1): # Just one! 
            ideal = ideal.reshape(1,-1)
            actual = actual.reshape(1,-1)
            r = Rot.align_vectors(ideal,actual)
            minval = r[1]
        newposits = r[0].apply(init_posits)
        conformerMolCpy.set_positions(newposits)
        rotatedConformer = conformerMolCpy
        temp_mol = io_molecule.convert_io_molecule(rotatedConformer)
        temp_mol.dist_sanity_checks()
        sane = temp_mol.dists_sane
    else: # One of the positions is messed up.
        rotatedConformer = conformerMolCpy
        sane = False
        minval = 100000
    if rot_coord_vect: # Apply a rotation around the coordination vector
        # Useful for sampling ligand rotations around coordinating sites,
        # Sampling more symmetries for the given set of binding sites.
        con_inds = [val[0] for val in ligcoordList]
        if len(con_inds) == 1:
            con_ind = con_inds[0]
            con_atom_posit = newposits[con_ind]
            actual = newposits - con_atom_posit #Center coordinates at con atom
            new_ideal = np.array((ideal.flatten()).tolist()) # Rotate to angle minimizing deviation from metal-coord atom line.
            ideal = np.array([new_ideal for x in range(len(actual))])
            r = Rot.align_vectors(ideal,actual)
            outnewposits = r[0].apply(actual)
            ideal = ideal/np.linalg.norm(ideal[0]) # Normalize for rotation around axis
            angle_rot = Rot.from_rotvec(ideal*rot_angle,degrees=True)
            outnewposits = angle_rot[0].apply(outnewposits)
            outnewposits = outnewposits + (con_atom_posit - outnewposits[con_ind]) # Reset to position
            conformerMolCpy.set_positions(outnewposits)
            rotatedConformer = conformerMolCpy
        else: # Determine central coordination vector to rotate all the points about.
            axis = ideal.sum(axis=0).flatten() # Take the sum of the vectors to be the rotation axis
            axis = axis/np.linalg.norm(axis) # Normalize for rotation around axis
            ivect = np.array([axis for x in range(len(newposits))])
            angle_rot = Rot.from_rotvec(ivect*rot_angle,degrees=True)
            outnewposits = angle_rot[0].apply(newposits)
            conformerMolCpy.set_positions(outnewposits)
            rotatedConformer = conformerMolCpy
    return (rotatedConformer, minval, sane)

def get_aligned_conformer(ligsmiles, ligcoordList, corecoordList, metal='Fe', 
                          add_angle_constraints=True, non_triangle=False,
                          rot_coord_vect=False, rot_angle=0, skip_mmff=False,
                          covrad_metal=None, vdwrad_metal=None, no_ff=False,
                          debug=False
                          ):
    """Uses distance geometry to get a random conformer.
        
    Parameters
    ----------
    ligsmiles : str
        smiles of ligand to analyze
    ligcoordList : list
        List of lists indicating [coordinating atom index, coordinating atom location on core]
    corecoordList : list
        List of lists inidicated connection sites by geometry 
    metal : str, optional
        Metal to be calculated -> by default, Fe
    covrad_metal : float, optional
        covalent radii of the metal, default None
    vdwrad_metal : float, optional
        vdw radii of the metal, default None
    no_ff : bool, optional
        Whether to do any ff cleaning at all.
    debug : bool, optional
        Print out debug messages, default False
        
    Returns
    -------
    outatoms : ase.Atoms
        ligand Atoms object without metal object aligned to coordination sites.
    minval : float
        rotational loss for the ligand atoms to the coordination sites.
    """
    Conf3D =  io_obabel.get_obmol_smiles(ligsmiles, addHydrogens=True)
    # Detect Cp rings - move before adding metal center.
    isCp, cp_rings, shared_coords = detect_cps(Conf3D, ligcoordList)
    catoms = [x[0] for x in ligcoordList]
    # Calc charges from total charge from OBmol
    init_charges_lig = np.zeros(Conf3D.NumAtoms())
    catoms = [x[0] for x in ligcoordList]
    dummy_metal = openbabel.OBAtom() # Add the dummy metal to the OBmol
    dummy_metal.SetAtomicNum(io_ptable.elements.index(metal)) # Match atomic number
    Conf3D.AddAtom(dummy_metal)
    for i in catoms:
        Conf3D.AddBond(int(i+1), Conf3D.NumAtoms(), 1)
    allcoords, anums, graph = io_obabel.get_OBMol_coords_anums_graph(Conf3D)
    bo_dict, atypes = io_obabel.get_OBMol_bo_dict_atom_types(Conf3D)
    init_charges_lig[0] = Conf3D.GetTotalCharge()
    cp_catoms = []
    if isCp:
        ligcoordList, corecoordList, cp_catoms, catoms = manage_cps(Conf3D, cp_rings, 
                                                 shared_coords, 
                                                 ligcoordList, 
                                                 corecoordList,
                                                 metal,
                                                 covrad_metal=covrad_metal
                                                 )
    # Check if any h-bound metal centers.
    if (not any([True for x in catoms if (anums[x] == 1)])): 
        # If not - run DG without hydrogens - add back during FF relaxation!
        Conf3D = io_obabel.get_obmol_smiles(ligsmiles, addHydrogens=False)
        dummy_metal = openbabel.OBAtom() # Add the dummy metal to the OBmol
        dummy_metal.SetAtomicNum(io_ptable.elements.index(metal)) # Match atomic number
        Conf3D.AddAtom(dummy_metal)
        for i in catoms:
            Conf3D.AddBond(int(i+1), Conf3D.NumAtoms(), 1)
        natoms = Conf3D.NumAtoms()
        allcoords, anums, graph = io_obabel.get_OBMol_coords_anums_graph(Conf3D)
        add_hydrogens=True
    else:
        natoms = Conf3D.NumAtoms()
        add_hydrogens=False
    core_vects = [corecoordList[x[1]] for x in ligcoordList]
    vdwrad = [io_ptable.rvdw[x] for x in anums]
    if isinstance(vdwrad_metal,float):
        vdwrad[-1] = vdwrad_metal
    cov1rad = [io_ptable.rcov1[x] for x in anums]
    if isinstance(covrad_metal,float):
        cov1rad[-1] = covrad_metal
    ml_dists = []
    if isCp: # Specifiy distances from geometry
        for cp_at in cp_catoms:
            tdist = np.linalg.norm([corecoordList[x[1]] for x in ligcoordList if x[0] == cp_at][0])
            ml_dists.append(tdist)
    for c_atom in [x for x in catoms if x not in cp_catoms]:
        ml_dists.append(cov1rad[-1] + cov1rad[c_atom]) 
    
    ############# Distance Geometry Section ##############
    # Here, I start with tighter constraints and loosen them to get to a final conformer.
    shape = get_ideal_angles(core_vects)
    status = False
    count = 0
    fail_gen = False
    LB, UB = get_bounds_matrix(allcoords, graph, natoms, 
                        catoms, shape, 
                        ml_dists, vdwrad, anums,
                        isCp=isCp, cp_catoms=cp_catoms,
                        add_angle_constraints=add_angle_constraints)
    while not status:
        try:
            tLB = LB.copy()
            tUB = UB.copy()
            D = metrize(tLB, tUB, natoms, non_triangle=non_triangle,debug=debug)
            D0 = get_cm_dists(D, natoms)
            G = get_metric_matrix(D, D0, natoms)
            L, V = get_3_eigs(G, natoms)
            X = np.dot(V, L)  # get projection
        except:
            X = np.array([np.nan])

        if np.any(np.isnan(X)): # If any are not nan continue!
            if count > 9:
                fail_gen = True
                status = True
            else:
                count += 1
        else:
            status = True
    if fail_gen:
        if debug:
            print('Trying More Dist Geom with 0.2 Bond Tol LB and Loosened Metal_Center Constraints.')
        status = False
        count = 0
        LB, UB = get_bounds_matrix(allcoords, graph, natoms, 
                        catoms, shape, 
                        ml_dists, vdwrad, anums, bond_tol=0.2, metal_center_lb_multiplier=0.9,
                        h_bond_tol=0.0, h_angle_tol=0.1,
                        isCp=isCp, cp_catoms=cp_catoms,
                        add_angle_constraints=add_angle_constraints)
        while not status:
            try:
                tLB = LB.copy()
                tUB = UB.copy()
                D = metrize(tLB, tUB, natoms, non_triangle=non_triangle,debug=debug)
                D0 = get_cm_dists(D, natoms)
                G = get_metric_matrix(D, D0, natoms)
                L, V = get_3_eigs(G, natoms)
                X = np.dot(V, L)  # get projection
            except:
                if debug:
                    print('Gen Exception.')
                X = np.array([np.nan])
                
            if np.any(np.isnan(X)): # If any are not nan continue!
                if debug:
                    # print('V: ',V, 'L: ', L, 'X: ',X)
                    print('DG iteration passing.')
                if count > 200:
                    fail_gen = True
                    status = True
                else:
                    count += 1
            else:
                status = True
                fail_gen = False # Worked!
    if fail_gen:
        if debug:
            print('Trying More Dist Geom with 0.2 Bond Tol LB, 0.2 Angle tol LB, and Loosened Metal_Center Constraints')
        status = False
        count = 0
        LB, UB = get_bounds_matrix(allcoords, graph, natoms, 
                        catoms, shape, 
                        ml_dists, vdwrad, anums, bond_tol=0.2,angle_tol=0.2, 
                        metal_center_lb_multiplier=0.8,
                        h_bond_tol=0.1, h_angle_tol=0.1,
                        isCp=isCp, cp_catoms=cp_catoms,
                        add_angle_constraints=add_angle_constraints)
        while not status:
            try:
                tLB = LB.copy()
                tUB = UB.copy()
                D = metrize(tLB, tUB, natoms, non_triangle=non_triangle,debug=debug)
                D0 = get_cm_dists(D, natoms)
                G = get_metric_matrix(D, D0, natoms)
                L, V = get_3_eigs(G, natoms)
                X = np.dot(V, L)  # get projection
            except:
                if debug:
                    print('Gen Exception.')
                X = np.array([np.nan])
                
            if np.any(np.isnan(X)): # If any are not nan continue!
                if debug:
                    # print('V: ',V, 'L: ', L, 'X: ',X)
                    print('DG iteration passing.')
                if count > 200:
                    fail_gen = True
                    status = True
                else:
                    count += 1
            else:
                status = True
                fail_gen = False # Worked!
    if fail_gen:
        if debug:
            print('Trying more Dist Geom Loosened Ca-M-Ca constraints')
        LB, UB = get_bounds_matrix(allcoords, graph, natoms, 
                catoms, shape, 
                ml_dists, vdwrad, anums, bond_tol=0.3,angle_tol=0.2, 
                metal_center_lb_multiplier=0.8, ca_angle_tol=0.2,
                h_bond_tol=0.1, h_angle_tol=0.2,
                isCp=isCp, cp_catoms=cp_catoms,
                add_angle_constraints=add_angle_constraints)
        status = False
        count = 0
        while not status:
            try:
                tLB = LB.copy()
                tUB = UB.copy()
                D = metrize(tLB, tUB, natoms, non_triangle=non_triangle, debug=debug)
                D0 = get_cm_dists(D, natoms)
                G = get_metric_matrix(D, D0, natoms)
                L, V = get_3_eigs(G, natoms)
                X = np.dot(V, L)  # get projection
            except:
                if debug:
                    print('Gen Exception.')
                X = np.array([np.nan])
                
            if np.any(np.isnan(X)): # If any are not nan continue!
                if debug:
                    # print('V: ',V, 'L: ', L, 'X: ',X)
                    print('DG iteration passing.')
                if count > 200:
                    fail_gen = True
                    status = True
                else:
                    count += 1
            else:
                status = True
                fail_gen = False # Worked!
    if fail_gen:
        if debug:
            print('Trying more Dist Geom with Dirty LB')
        LB, UB = get_bounds_matrix(allcoords, graph, natoms, 
                                    catoms, shape, 
                                    ml_dists, vdwrad, anums, bond_tol=0.3,angle_tol=0.3, 
                                    metal_center_lb_multiplier=0.6, ca_angle_tol=0.3,
                                    h_bond_tol=0.1, h_angle_tol=0.3,
                                    isCp=isCp, cp_catoms=cp_catoms,
                                    add_angle_constraints=add_angle_constraints)
        status = False
        count = 0
        while not status:
            try:
                tLB = LB.copy()
                tUB = UB.copy()
                D = metrize(tLB, tUB, natoms, non_triangle=non_triangle, debug=debug)
                D0 = get_cm_dists(D, natoms)
                G = get_metric_matrix(D, D0, natoms)
                L, V = get_3_eigs(G, natoms)
                X = np.dot(V, L)  # get projection
            except:
                if debug:
                    print('Gen Exception.')
                X = np.array([np.nan])
                
            if np.any(np.isnan(X)): # If any are not nan continue!
                if debug:
                    # print('V: ',V, 'L: ', L, 'X: ',X)
                    print('DG iteration passing.')
                if count > 200:
                    fail_gen = True
                    status = True
                else:
                    count += 1
            else:
                status = True
                fail_gen = False # Worked!

    ############# END Distance Geometry Section ##############
    ############# Start FF Relaxation Section ##############
 
    if not fail_gen:
        x = np.reshape(X, 3*natoms)
        res1 = optimize.fmin_cg(distance_error, x, fprime=dist_error_gradient,
                                gtol=0.1, args=(LB, UB, natoms), disp=0)
        if np.any(np.isnan(res1)):
            if debug:
                print('Dist Geom produced Nan. - Skipping!')
            Conf3D_out = io_obabel.convert_obmol_ase(Conf3D, add_hydrogens=add_hydrogens)
            final_relax = False
            fail = True
            tligcoordList = ligcoordList.copy()
        else:
            X = np.reshape(res1, (natoms, 3))
            if no_ff:
                Conf3D_out, fail, final_relax, tligcoordList = nonclean_conformation_ff(X, 
                                Conf3D, catoms, shape, 
                                graph, anums, ligcoordList,
                                add_angle_constraints=add_angle_constraints,
                                isCp=isCp,
                                cp_catoms=cp_catoms,
                                skip_mff=skip_mmff,add_hydrogens=add_hydrogens
                                )
            else:
                Conf3D_out, fail, final_relax, tligcoordList = clean_conformation_ff(X, 
                                                Conf3D, catoms, shape, 
                                                graph, anums, ligcoordList,
                                                add_angle_constraints=add_angle_constraints,
                                                isCp=isCp,
                                                cp_catoms=cp_catoms,
                                                skip_mff=skip_mmff,add_hydrogens=add_hydrogens,
                                                debug=debug
                                                )

            # Set charges from total charge from OBmol
            Conf3D_out.set_initial_charges(init_charges_lig)
    else:
        if debug:
            print('Dist Geom Failed entirely!')
        Conf3D_out = io_obabel.convert_obmol_ase(Conf3D,add_hydrogens=add_hydrogens)
        final_relax = True
        fail = True
        tligcoordList = ligcoordList.copy()

    ############# END FF Relaxation Section ##############

    if (not fail): # Catch FF optimization failures
        (outatoms, minval, sane) = set_position_align(Conf3D_out, 
                                                tligcoordList, 
                                                corecoordList, 
                                                isCp=isCp,
                                                debug=debug,
                                                rot_coord_vect=rot_coord_vect, 
                                                rot_angle=rot_angle
                                                )
        crowding_penalty = 0
        if sane: # Perform check to see if metal is extra crowded.
            cov1metal = cov1rad[-1]
            if add_hydrogens:
                LB_dists = np.array([io_ptable.rcov1[x] for x in outatoms.get_atomic_numbers()]) + cov1metal
            else:
                LB_dists = np.array(cov1rad[0:-1]) + cov1metal
            coords = outatoms.positions
            dists = np.linalg.norm(coords,axis=1)
            max_catom_dist = np.max(dists[catoms])
            if np.any(dists < LB_dists*0.5):
                sane = False
            # Add penalty against crowded conformations.
            crowding_penalty += len(np.where(dists < LB_dists*0.8)[0])
            # Super penalize conformers with any atoms closer than the max connecting atom distance.
            if np.any(np.delete(dists,catoms) < max_catom_dist):
                crowding_penalty += 100
            ### Potentially add penalty for crowding other binding sites.
    else: # Save conformation - but set it's rotational error high.
        (outatoms, minval, sane) = Conf3D_out, 10000, False
        crowding_penalty = 1000
        if debug:
            print('Failed Relaxation!')
    minval += crowding_penalty
    return outatoms, minval, sane, final_relax, bo_dict, atypes, tligcoordList

def find_conformers(ligsmiles, ligcoordList, corecoordList, metal='Fe', nconformers=3,
                    ligtype=None, skip_mmff=False, covrad_metal=None, vdwrad_metal=None,
                    no_ff=False, debug=False
                    ):
    """find_conformers 
    Take in ligsmiles and generate N different conformers 

    Parameters
    ----------
    ligsmiles : str
        ligand smiles string
    ligcoordList : list
        ligand coord list
    corecoordList : list
        core coord list
    metal : str, optional
        metal identity, by default 'Fe'
    nconformers : int, optional
        number of conformers to test, by default 5
    ligtype : str, optional
        ligand type for generation
    skip_mmff : bool, optional
        Skip MMFF relaxations, by default False
    covrad_metal : bool, optional
        covalent radii of the metal, by defualt None
    vdwrad_metal : bool, optional
        vdw radii of the metal, by default None
    no_ff : bool, optional
        Entirely skip FF ralaxations (get distance geomtetry structure), default False
    debug : bool, optional
        Print out debug information?, default False

    Returns
    -------
    conf_list : list
        list of ase atoms objects.
    val_list : list
        list of values of the conformers to match the rotation of the coordination sites.
    rot_list : list
        list of rotations applied the ligands in degrees.
    """
    if debug:
        print('DEBUG LIGANDS:',ligsmiles, ligcoordList, corecoordList, metal) #for debug
    conf_list = []
    val_list = []
    tligcoordList_out = []
    rot_list = []
    
    if debug:
        seeds = [np.random.randint(1,100) for x in range(nconformers)]
        print('Init seeds (pre-ligand generation): ', seeds)
    OBmol_lig = io_obabel.get_obmol_smiles(ligsmiles)

    if (OBmol_lig.NumAtoms() < 4): # Limit smaller molecules conformers (not useful)
        conf, val, sane, final_relax, bo_dict, atypes, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                         corecoordList, 
                                         metal=metal,
                                         skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                         vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        else:
            if debug:
                print('Failed sanity checks after rotation!')
    elif ('cis' in ligtype) or ('mer' in ligtype) or ('fac' in ligtype):
        if debug:
            print('Generating cis/mer/facs')
        conf, val, sane, final_relax, bo_dict, atypes, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                         corecoordList, 
                                         metal=metal,
                                         skip_mmff=skip_mmff,covrad_metal=covrad_metal,
                                         vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        else:
            if debug:
                print('Failed sanity checks after rotation!')
        if debug:
            print('Generating cis/mer/facs.')
        conf, val, sane, final_relax, bo_dict, atypes, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                         corecoordList, 
                                         metal=metal,
                                         skip_mmff=True, covrad_metal=covrad_metal,
                                         vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        else:
            if debug:
                print('Failed sanity checks after rotation!')
        
        # Limit conformers for bidentate cis/tri_mer/tri_fac -> generate rotated/flipped versions!!!
        # for i in range(1):
        conf, val, sane, final_relax, bo_dict, atypes, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                        corecoordList, 
                                        metal=metal, 
                                        skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                         vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        else:
            if debug:
                print('Failed sanity checks after rotation!')
        
        # Add N+1 without the explicit angle constraints
        conf, val, sane, final_relax, _ , _, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                            corecoordList, 
                                            metal=metal, 
                                            add_angle_constraints=False,
                                            skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                            vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        
        # Add N+2 without the triangle angle constraint for metal (encouraging different conformers for multidentate)
        conf, val, sane, final_relax, _, _, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                            corecoordList, 
                                            metal=metal, 
                                            add_angle_constraints=True,
                                            non_triangle=True,
                                            skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                            vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)

        # Add N+2 without the triangle angle constraint for metal (encouraging different conformers for multidentate)
        conf, val, sane, final_relax, _, _, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                            corecoordList, 
                                            metal=metal,  
                                            add_angle_constraints=False,
                                            non_triangle=True,
                                            skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                            vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
    else:
        for i in range(nconformers):
            conf, val, sane, final_relax, bo_dict, atypes, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                            corecoordList, 
                                            metal=metal, 
                                            skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                            vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
            if sane:
                conf_list.append(conf)
                val_list.append(val)
                tligcoordList_out.append(tligcoordList)
                rot_list.append(0)
            else:
                if debug:
                    print('Failed sanity checks after rotation!')

        # Add N+1 without the explicit angle constraints
        conf, val, sane, final_relax, _ , _, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                            corecoordList, 
                                            metal=metal, 
                                            add_angle_constraints=False,
                                            skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                            vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        
        # print('Add conformer with no MMFF relaxation')
        conf, val, sane, final_relax, bo_dict, atypes, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                         corecoordList, 
                                         metal=metal, 
                                         skip_mmff=True, covrad_metal=covrad_metal,
                                         vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        else:
            if debug:
                print('Failed sanity checks after rotation!')
        
        # Add N+2 with the triangle angle constraint for metal (encouraging different conformers for multidentate)
        conf, val, sane, final_relax, _, _, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                            corecoordList, 
                                            metal=metal, 
                                            add_angle_constraints=True,
                                            non_triangle=True,
                                            skip_mmff=skip_mmff, covrad_metal=covrad_metal,
                                            vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)
        
        # Add N+2 without the triangle angle constraint for metal (encouraging different conformers for multidentate)
        conf, val, sane, final_relax, _, _, tligcoordList = get_aligned_conformer(ligsmiles, ligcoordList, 
                                            corecoordList, 
                                            metal=metal, 
                                            add_angle_constraints=False,
                                            non_triangle=True,
                                            skip_mmff=skip_mmff, 
                                            covrad_metal=covrad_metal,
                                            vdwrad_metal=vdwrad_metal,no_ff=no_ff,debug=debug)
        if sane:
            conf_list.append(conf)
            val_list.append(val)
            tligcoordList_out.append(tligcoordList)
            rot_list.append(0)

    if (len(ligcoordList) == 1) and (len(conf_list) > 0) and (OBmol_lig.NumAtoms()>1): # Add aligned monodentate and 45 and 90-degree rotated conformers
        conf = conf_list[0]
        rotatedConformer, _ , _ = set_position_align(conf, tligcoordList_out[0], corecoordList, isCp=False, debug=debug,
                       rot_coord_vect=True, rot_angle=0)
        conf_list.append(rotatedConformer)
        val_list.append(val_list[0])
        rot_list.append(0)
        tligcoordList_out.append(tligcoordList_out[0])
        if OBmol_lig.NumAtoms() > 2: # For 3 add rotation versions of the ligands
            rotatedConformer, _ , _ = set_position_align(conf, tligcoordList_out[0], corecoordList, isCp=False, debug=debug,
            rot_coord_vect=True, rot_angle=45)
            conf_list.append(rotatedConformer)
            rot_list.append(45)
            val_list.append(val_list[0])
            tligcoordList_out.append(tligcoordList_out[0])
            rotatedConformer, _ , _ = set_position_align(conf, tligcoordList_out[0], corecoordList, isCp=False, debug=debug,
            rot_coord_vect=True, rot_angle=90)
            rot_list.append(90)
            conf_list.append(rotatedConformer)
            val_list.append(val_list[0])
            tligcoordList_out.append(tligcoordList_out[0])
    # Add in rotated 180 degree rotated versions of the generated conformers
    elif ('mer' in ligtype) or ('cis' in ligtype) and (len(conf_list) > 0):
        if debug:
            print('Adding rotated mer/cis.')
        conf_list_cp = conf_list.copy()
        for i,conf_copy in enumerate(conf_list_cp):
            rotatedConformer, _ , _ = set_position_align(conf_copy, tligcoordList_out[i], corecoordList, isCp=False, debug=debug,
                rot_coord_vect=True, rot_angle=180)
            conf_list.append(rotatedConformer)
            rot_list.append(180)
            val_list.append(val_list[i])
            tligcoordList_out.append(tligcoordList_out[i])
    # Add in rotated 120 and 240 degree rotated versions of the generated conformers.
    elif ('fac' in ligtype):
        if debug:
            print('Adding rotated fac ligands.')
        conf_list_cp = conf_list.copy()
        for i,conf_copy in enumerate(conf_list_cp):
            rotatedConformer, _ , _ = set_position_align(conf_copy, tligcoordList_out[i], corecoordList, isCp=False, debug=debug,
                rot_coord_vect=True, rot_angle=120)
            conf_list.append(rotatedConformer)
            rot_list.append(120)
            val_list.append(val_list[i])
            tligcoordList_out.append(tligcoordList_out[i])
            rotatedConformer, _ , _ = set_position_align(conf_copy, tligcoordList_out[i], corecoordList, isCp=False, debug=debug,
                rot_coord_vect=True, rot_angle=240)
            conf_list.append(rotatedConformer)
            rot_list.append(240)
            val_list.append(val_list[i])
            tligcoordList_out.append(tligcoordList_out[i])

    if debug:
        seeds = [np.random.randint(1,100) for x in range(nconformers)]
        print('Final seeds (post-ligand generation): ', seeds)
        
    return conf_list, val_list, tligcoordList_out, final_relax, bo_dict, atypes, rot_list