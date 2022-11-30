"""
Core geometry information mining.

Geomety class contains all the information of possible cores and ligands mapping to cores.

Routines for mapping user-defined cores/ligands encoded here!

Developed by Michael Taylor and Jan Janssen
"""

import itertools
import numpy as np
import pandas as pd
import os
import architector.io_ptable as io_ptable
import architector.geometries as geo
import warnings
import architector.io_molecule as io_molecule

warnings.filterwarnings('error')

np.seterr(all='warn')


def get_lig_ref_df():
    """get_lig_ref_df 
    Pull the ligand angle references.

    Returns
    -------
    ref_df : dict
        dictionary with {'type':angles(List)}
    """
    filepath = os.path.abspath(os.path.join(__file__, "..", "data", "ligtype_angle_reference.csv"))
    ref_df = pd.read_csv(filepath)
    return ref_df


def get_angle(coord1, coord2, coord3):
    """get_angle 

    Parameters
    ----------
    coord1 : list
        coordinates of point 1
    coord2 : list
        coordinates of point 2
    coord3 : list
        coordinates of point 3

    Returns
    -------
    angle : float
        angle between 1-2-3 (2 at center)
    """
    if isinstance(coord1, list):
        coord1 = np.array(coord1)
    if isinstance(coord2, list):
        coord2 = np.array(coord2)
    if isinstance(coord3, list):
        coord3 = np.array(coord3)
    v1 = coord1-coord2
    v2 = coord3-coord2
    try:
        angle = np.degrees(np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
        if np.isnan(angle):
            if np.any(np.isclose(v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2),0)):
                return 180.0
            else:
                return 0.0
        return angle
    except Warning: # Catch cases where angles are too close (parallel/antiparallel)
        if np.any(np.isclose(v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2),0)):
            return 180.0
        else:
            return 0.0
    

def check_intercalation(coordat_list, all_coordats):
    """check_intercalation 
    Check for any intercalating points between selected indices

    Parameters
    ----------
    coordat_list : list
        list of coordinating atoms selected for specific geometry
    all_coordats : list
        all possible coordinating atoms for specific geometry

    Returns
    -------
    intercalating : bool
        whether or not any of the remaining coord atoms at the surface of the metal
        fall between the potential selected coordinating atoms for the ligand
    """
    metal_coords = [0.0, 0.0, 0.0]
    coordats = np.array(coordat_list)
    if len(coordats) > 1:
        all_coordats = np.array(all_coordats)
        test_coordats = []
        for coord in all_coordats:  # Remove matched coordatoms from full list.
            good = True
            for coordat in coordats:
                if np.linalg.norm(coord-coordat) < 1e-12:
                    good = False
            if good:
                test_coordats.append(coord)
        test_coordats = np.array(test_coordats)
        intercalating = False
        for pair in itertools.combinations(coordats, 2):  # Take pairs of coord atoms
            if get_angle(pair[0], metal_coords, pair[1]) < 121:  # If the angle is less than 121 check for intercalation
                midvect = (pair[0] + pair[1])  # Take the midpoint vector between the two cord points
                for coordat in test_coordats:  # Test the remaining coordintion sites
                    if get_angle(midvect, metal_coords, coordat) < 30:
                        # If any of the points are within 30 degrees of the midpoint
                        # Flag as intercalating
                        intercalating = True
    else:
        intercalating = False
    return intercalating

denticity_combinations_dict = {0:1, 1:2, 3:3, 6:4, 10:5, 15:6, 21:7, 28:8, 36:9}

def calc_angles_struct(structure):
    """ Read in structure to assign ligand to right shape

    structure : mol2string
    """
    if isinstance(structure,list):
        lig_angles = np.array(structure)[np.argsort(angs)[::-1]]
    else:
        mol = io_molecule.convert_io_molecule(structure)
        if len(mol.graph) == 0:
            print('Creating imputed molecular graph! May be untrustworthy.')
            mol.create_BO_dict()
        mets = mol.find_metals()
        lig_angles = []
        if len(mets) == 1:
            coordats = np.nonzero(mol.graph[mets[0]])[0]
            if len(coordats) == 1:
                lig_angles = []
            else:
                coords = mol.ase_atoms.get_positions()
                angs = [
                    get_angle(coords[x[0]],coords[mets[0]],coords[x[1]]) for x in itertools.combinations(coordats,2)
                    ]
                angs = np.array(angs)[np.argsort(angs)[::-1]] # Add angles
                lig_angles += angs.tolist() # Add sorted angles as features
        else:
            print('Warning: User ligand input without metal for refernce on interatomic angles. \
                    Please pass a structure with a metal for user ligand generation.')
    lig_angles += [0.0] * (36-len(lig_angles)) # Pad with zeros
    n_ca_m_ca_angles = len(np.nonzero(lig_angles)[0])
    denticity = denticity_combinations_dict[n_ca_m_ca_angles]
    return {'user_lig':np.array(lig_angles)}, denticity

class Geometries:
    """ geometries Dataclass containing information and corecoordLists for all
    most frequent molecular geometries:
    https://en.wikipedia.org/wiki/Capped_square_antiprismatic_molecular_geometry 
    """
    def __init__(self, usercore=False):
        """__init__ 
        """
        self.geometry_dict = {
            f: getattr(geo, f)
            for f in dir(geo)
            if not f.endswith("__")
        }

        # Add in user's core information
        if isinstance(usercore,(list,np.ndarray)):
            self.geometry_dict.update({'user_core':usercore})

        self.geo_cn_dict = {
            k: len(v)
            for k, v in self.geometry_dict.items()
        }

        self.cn_geo_dict = {
            k: []
            for k in set(self.geo_cn_dict.values())
        }
        for k, v in self.geo_cn_dict.items():
            self.cn_geo_dict[v].append(k)

        self.ambiguous_goes = [
            'bicapped_trigonal_prismatic',
            'capped_trigonal_prismatic',
            'capped_octahedral',
            'capped_square_antiprismatic'
        ]

        self.trans_geos = [
            'linear',
            't_shaped',
            'square_planar',
            'seesaw',
            'square_pyramidal',
            'trigonal_bipyramidal',
            'octahedral',
            'pentagonal_bipyramidal',
            'hexagonal_bipyramidal',
            'axial_bicapped_trigonal_prismatic',
            'axial_bicapped_hexagonal_planar',
            'penta_bi_capped_pyramidal',
            'senrag_comp_1'
        ]

        self.liglist_geo_map_dict = None

    def rescale_refgeos(self, core_atom, rcovmetal=None):
        """rescale_refgeos 
        rescale the geometry dictionary to closer match the actual atomic radii
        matters for non-perfectly symmetric core geometries!

        Parameters
        ----------
        core_atom : str
            chemical symbol of core atom
        rcovmetal : float, optional
            Covalent radii of the metal, defaults to values in io_ptable.rcov1
        """
        if isinstance(rcovmetal,float):
            interatomic_dist_ideal = rcovmetal + io_ptable.rcov1[io_ptable.elements.index('O')]
        else:
            interatomic_dist_ideal = \
                io_ptable.rcov1[io_ptable.elements.index(core_atom)] + io_ptable.rcov1[io_ptable.elements.index('O')]
        geodict_copy = self.geometry_dict.copy()
        for key in self.geometry_dict.keys():
            geo_dict = self.geometry_dict[key]
            dists = np.linalg.norm(geo_dict, axis=1)
            avg_dist = np.mean(dists)
            re_scale = interatomic_dist_ideal/avg_dist
            scaled_geo = []
            for xyz in geo_dict:
                out = [xyz[0]*re_scale, xyz[1]*re_scale, xyz[2]*re_scale]
                scaled_geo.append(out)
            geodict_copy.update({key: scaled_geo})
        self.geometry_dict = geodict_copy

    def get_lig_ref_inds_dict(self, core_atom, coreTypes, userlig=False, rcovmetal=None): 
        """get_lig_ref_inds_dict 
        Map the reference ligand angle information onto possible
        core geometries!

        Parameters
        ----------
        core_atom : str
            metal atomic symbol
        coreTypes : list
            list of core coordination types (e.g. ['octahedral','tetrahedral'...])
        userlig : bool/list/molecule.
            list of reference ligands
        rcovmetal : float
            radii to be imposed for metal
        """
        df = get_lig_ref_df()  # Get ligand angle reference information

        # Translate reference data into ligands info here.
        self.ligType_cn_dict = dict()
        for i,row in df.iterrows():
            self.ligType_cn_dict.update({row['ligtype']:int(row['denticity'])})
        
        self.cn_ligType_dict = {
            k: []
            for k in set(self.ligType_cn_dict.values())
        }
        for k, v in self.ligType_cn_dict.items():
            self.cn_ligType_dict[v].append(k)

        self.rescale_refgeos(core_atom,rcovmetal=rcovmetal)
        # Add userlig information here - handle either reference ligand/metal geometry
        if not isinstance(userlig,bool):
            userlig, denticities = calc_angles_struct(userlig)
            count = 0
            for key,val in userlig.items():
                df.loc[len(df),:] = list(val)
                self.ligType_cn_dict.update({'user_lig'+str(count):denticities[count]})
                tlist = self.cn_ligType_dict[denticities[count]]
                tlist.append('user_lig'+str(count))
                self.cn_ligType_dict.update({denticities[count]:tlist})
                count += 1
        mean_angle_labels = [x for x in df.columns.values if '_mean' in x]  # Mean angle!
        metal_coords = [0.0, 0.0, 0.0]
        outdict_rows = []
        labels = []
        possible_geos = coreTypes

        for i, row in df.iterrows():
            outdict = dict()
            outdict_losses = dict()
            labels.append(row['ligtype'])
            cn_min = row['denticity']
            mean_angles = row[mean_angle_labels].values
            max_angles = mean_angles + np.ones(36)*30  # Use looser criterion to get more points.
            min_angles = mean_angles - np.ones(36)*30
            for core_type in possible_geos:  # Define possible coordination environments and selected indices
                # Rule out pairs or sets where additional points fall between the selected indices.
                coordat_locs = self.geometry_dict[core_type]
                all_coordat_inds = [x for x in itertools.combinations(range(len(coordat_locs)), cn_min)]
                all_coordat_combs = []
                for x in all_coordat_inds:
                    tmp = []
                    for k in x:
                        tmp.append(coordat_locs[k])
                    all_coordat_combs.append(tmp)
                for j, coordat_comb in enumerate(all_coordat_combs):
                    angs = [
                        get_angle(x[0], metal_coords, x[1]) for x in itertools.combinations(coordat_comb, 2)
                        ]
                    angs = np.array(angs)[np.argsort(angs)[::-1]]  # sort angles largest first!
                    if cn_min < 9:  # pad angles with zeros
                        angs = np.pad(angs, (0, 36-len(angs)), 'constant')
                    if np.all(angs <= max_angles) & np.all(angs >= min_angles):
                        loss = np.mean(np.abs(angs - mean_angles))  # MAE
                        # Check for intercalating coordination points!!! -> don't use these!!!
                        intercalating = check_intercalation(coordat_comb, coordat_locs)
                        if (core_type in outdict) and (not intercalating):
                            temp = outdict[core_type].copy()
                            temp.append(all_coordat_inds[j])
                            outdict.update({core_type: temp})
                            tmp_losses = outdict_losses[core_type].copy()
                            tmp_losses.append(loss)
                            outdict_losses.update({core_type: tmp_losses})  # Save losses
                        elif not intercalating:
                            outdict.update({core_type: [all_coordat_inds[j]]})
                            outdict_losses.update({core_type: [loss]})
                # Sort outdict by losses
                for key in outdict.keys():
                    tmp_inds = outdict[key].copy()
                    tmp_losses = outdict_losses[key]
                    order = np.argsort(tmp_losses)  # Smallest loss first
                    out = np.array(tmp_inds)[order]
                    out = out.tolist()
                    outdict[key] = out
            outdict_rows.append(outdict)

        for dent in [10,11,12]: # For higher denticity ligands set them to all possible combinations of locations
            labels.append(str(dent))
            cn_min=dent
            outdict = dict()
            for core_type in possible_geos:
                coordat_locs = self.geometry_dict[core_type]
                all_coordat_inds = [x for x in itertools.combinations(range(len(coordat_locs)), cn_min)]
                outdict.update({core_type:all_coordat_inds})
            outdict_rows.append(outdict)

        total_dict = dict(zip(labels, outdict_rows))
        # Assign sandwich to tri_fac geometries!
        total_dict['sandwich'] = total_dict['tri_fac']
        self.liglist_geo_map_dict = total_dict


def calc_all_coord_atom_angles(core_geo):
    """calc_all_coord_atom_angles 
    Get all of the L-M-L angles for a core geometry. 

    Parameters
    ----------
    core_geo : np.ndarray
        core geometry with coordinates of all coordinating atoms.

    Returns
    -------
    np.ndarray
        sorted array of L-M-L angles
    """
    origin = np.array((0,0,0))
    angles = []
    for a,b in itertools.combinations(core_geo,2):
        angles.append(get_angle(a,origin,b))
    return np.array(sorted(angles))
        

def classify_metal_geo_type(mol2string,metal_ind=None):
    """classify_metal_geo_type calculate the actual geometry of the metal centers

    Parameters
    ----------
    mol2string : str
        mol2string with metals
    metal_ind : int or list, optional
        the metal indices, by default None

    Returns
    -------
    metal_center_geos : dict or list
        metal center geometries present in the mol2string.
    """
    molecule = io_molecule.convert_io_molecule(mol2string)
    if metal_ind is None:
        metal_ind = [i for i,x in enumerate(molecule.ase_atoms.get_chemical_symbols()) if (x in io_ptable.all_metals)]
    elif isinstance(metal_ind,int):
        metal_ind = [metal_ind]
    elif isinstance(metal_ind,(np.ndarray,list)):
        metal_ind = metal_ind 
    else:
        raise ValueError('Unrecognized type for metal_ind')
    geo_dict = Geometries()
    if len(metal_ind) == 0:
        raise ValueError('No metal or ind passed in this molecule.')
    elif len(metal_ind) > 1: # Look at every metal center
        metal_center_geos = []
        for metal_indx in metal_ind: 
            tmpdict = dict()
            neighs = np.nonzero(np.ravel(molecule.graph[metal_indx]))[0]
            if (len(neighs) < 13) and (len(neighs) != 11):
                coord_at_positions = molecule.ase_atoms.positions[neighs] - molecule.ase_atoms.positions[metal_indx]
                act_geo_vect = calc_all_coord_atom_angles(coord_at_positions)
                ref_geo_labels = geo_dict.cn_geo_dict[len(neighs)]
                ref_geos = [calc_all_coord_atom_angles(geo_dict.geometry_dict[x]) for x in ref_geo_labels]
                mae_losses = [np.mean(np.abs(act_geo_vect - x)) for x in ref_geos] # Calc MAE loss between interatomic angles
                sort_order = np.argsort(mae_losses)
                m_geo_type = ref_geo_labels[np.argmin(mae_losses)]
                tmpdict['metal'] = molecule.ase_atoms.get_chemical_symbols()[metal_indx]
                tmpdict['metal_ind'] = metal_indx
                tmpdict['metal_geo_type'] = m_geo_type
                tmpdict['mae_angle_loss'] = mae_losses[np.argmin(mae_losses)]
                if len(sort_order) > 1:
                    tmpdict['confidence'] = 1 - tmpdict['mae_angle_loss'] / mae_losses[sort_order[1]]
                else:
                    tmpdict['confidence'] = 1
                tmpdict['classification_dict'] = {ref_geo_labels[i]:mae_losses[i] for i in sort_order}
            else: 
                tmpdict['metal'] = molecule.ase_atoms.get_chemical_symbols()[metal_indx]
                tmpdict['metal_ind'] = metal_indx
                tmpdict['metal_geo_type'] = len(neighs)
            metal_center_geos.append(tmpdict)
    else: # Just calculate geometry for one metal center.
        metal_center_geos = dict()
        metal_indx = metal_ind[0]
        neighs = np.nonzero(np.ravel(molecule.graph[metal_indx]))[0]
        if len(neighs) < 13 and (len(neighs) != 11):
            coord_at_positions = molecule.ase_atoms.positions[neighs] - molecule.ase_atoms.positions[metal_indx]
            act_geo_vect = calc_all_coord_atom_angles(coord_at_positions)
            ref_geo_labels = geo_dict.cn_geo_dict[len(neighs)]
            ref_geos = [calc_all_coord_atom_angles(geo_dict.geometry_dict[x]) for x in ref_geo_labels]
            mae_losses = [np.mean(np.abs(act_geo_vect - x)) for x in ref_geos]
            sort_order = np.argsort(mae_losses)
            m_geo_type = ref_geo_labels[np.argmin(mae_losses)]
            metal_center_geos['metal'] = molecule.ase_atoms.get_chemical_symbols()[metal_indx]
            metal_center_geos['metal_ind'] = metal_indx
            metal_center_geos['metal_geo_type'] = m_geo_type
            metal_center_geos['mae_angle_loss'] = mae_losses[np.argmin(mae_losses)]
            if len(sort_order) > 1:
                metal_center_geos['confidence'] = (mae_losses[sort_order[1]]-metal_center_geos['mae_angle_loss'])/metal_center_geos['mae_angle_loss']
            else:
                metal_center_geos['confidence'] = 1
            metal_center_geos['classification_dict'] = {ref_geo_labels[i]:mae_losses[i] for i in sort_order}
        else: 
            metal_center_geos['metal'] = molecule.ase_atoms.get_chemical_symbols()[metal_indx]
            metal_center_geos['metal_ind'] = metal_indx
            metal_center_geos['metal_geo_type'] = len(neighs)
    return metal_center_geos