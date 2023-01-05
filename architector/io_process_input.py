"""
Code for parsing and processing user input dictionaries!

Developed by Michael Taylor
"""

import architector
import architector.io_ptable as io_ptable
import architector.io_core as io_core
import architector.io_lig as io_lig
import architector.io_obabel as io_obabel
import architector.io_molecule as io_molecule
import numpy as np
import pandas as pd
import uuid
import copy
import os
import mendeleev
from ase.db import connect

def isnotebook():
    """isnotebook
    hacky function to determine if exection is inside a jupyer notebook or ipython shell

    from: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    returns bool
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def assign_ligType_bruteforce(core_geo_class, ligsmiles, ligcoords, metal, covrad_metal=None, vdwrad_metal=None):
    """assign_ligType_bruteforce
    Assign the ligtype based on best average rotational loss when being assigned 
    to coordination sites corresponding to the ligand type.

    Parameters
    ----------
    core_geo_class : architector.io_core.Geometries
        core geometries class with ligandType maps already calculated.
    ligsmiles : str
        ligand smiles
    ligcoords : list
        1D ligand coordinating atom list
    metal : str
        metal
    covrad_metal : float
        covalent radii of the metal, default None
    vdwrad_metal : float
        vdw radii of the metal, default None

    Returns
    -------
    ligType : str
        The lowest-rotational loss ligand type.

    Raises
    ------
    ValueError
        Will flag if this ligand/metal/core geometry can't generate any valid conformations.
    """
    ## First check for cp rings -> all indices in shared ring.
    OBmol = io_obabel.get_obmol_smiles(ligsmiles)
    rings = OBmol.GetSSSR()
    is_cp = False
    for ring in rings:
        if all(ring.IsInRing(x+1) for x in ligcoords) and (len(ligcoords) > 2) and (ring.IsAromatic()):
            is_cp = True
    if is_cp:
        return 'sandwich'
    elif len(ligcoords) > 9: # 10-12 are saved under these monikers.
        return str(len(ligcoords))
    else:
        lig_denticity = len(ligcoords)
        possible_Ligtypes = core_geo_class.cn_ligType_dict[lig_denticity]
        rot_vals = []
        possible_saves = []
        print('ligType not specified for {} - testing ligand placement to determine ligType!'.format(ligsmiles))
        print('Warning: can take a while depending on the size of the ligand.')
        # Test by attempting to place the ligand on pre-calculated cores!
        for ligtype in possible_Ligtypes:
            rot_vals.append([])
            possible_saves.append(False)
            for coreType in core_geo_class.liglist_geo_map_dict[ligtype]:
                corecoordList = core_geo_class.geometry_dict[coreType]
                # Pick first one.
                core_cons = core_geo_class.liglist_geo_map_dict[ligtype][coreType][0]
                tmp_ligList = []
                for i in range(len(ligcoords)):
                    tmp_ligList.append([ligcoords[i],core_cons[i]])
                # Test mapping on just single ligand conformation.
                # Minval represents the loss
                _, minval, sane, _, _, _, _ = io_lig.get_aligned_conformer(
                            ligsmiles, tmp_ligList, corecoordList, metal=metal, 
                            covrad_metal=covrad_metal, vdwrad_metal=vdwrad_metal)
                if sane:
                    rot_vals[-1].append(minval)
                    possible_saves[-1] = True
                else:
                    _, minval, sane, _, _, _, _ = io_lig.get_aligned_conformer(
                            ligsmiles, tmp_ligList, corecoordList, metal=metal, 
                            covrad_metal=covrad_metal, vdwrad_metal=vdwrad_metal)
        scores = [np.mean(x) for x in rot_vals]
        min_ind = np.argsort(scores)[0]
        if not possible_saves[min_ind]: # Try with larger radii
            tcovrad_metal = io_ptable.rcov1[io_ptable.elements.index(metal)] * 1.3
            tvdwrad_metal = io_ptable.rvdw[io_ptable.elements.index(metal)] * 1.3
            rot_vals = []
            possible_saves = []
            print('ligType not specified for {} - testing ligand placement to determine ligType!'.format(ligsmiles))
            print('Warning: can take a while depending on the size of the ligand.')
            # Test by attempting to place the ligand on pre-calculated cores!
            for ligtype in possible_Ligtypes:
                rot_vals.append([])
                possible_saves.append(False)
                for coreType in core_geo_class.liglist_geo_map_dict[ligtype]:
                    corecoordList = core_geo_class.geometry_dict[coreType]
                    # Pick first one.
                    core_cons = core_geo_class.liglist_geo_map_dict[ligtype][coreType][0]
                    tmp_ligList = []
                    for i in range(len(ligcoords)):
                        tmp_ligList.append([ligcoords[i],core_cons[i]])
                    # Test mapping on just single ligand conformation.
                    # Minval represents the loss
                    _, minval, sane, _, _, _, _ = io_lig.get_aligned_conformer(
                                ligsmiles, tmp_ligList, corecoordList, metal=metal, 
                                covrad_metal=tcovrad_metal, vdwrad_metal=tvdwrad_metal)
                    if sane:
                        rot_vals[-1].append(minval)
                        possible_saves[-1] = True
            scores = [np.mean(x) for x in rot_vals]
            min_ind = np.argsort(scores)[0]
        if not possible_saves[min_ind]: # Try with smaller radii
            tcovrad_metal = io_ptable.rcov1[io_ptable.elements.index(metal)] * 0.8
            tvdwrad_metal = io_ptable.rvdw[io_ptable.elements.index(metal)] * 0.8
            rot_vals = []
            possible_saves = []
            print('ligType not specified for {} - testing ligand placement to determine ligType!'.format(ligsmiles))
            print('Warning: can take a while depending on the size of the ligand.')
            # Test by attempting to place the ligand on pre-calculated cores!
            for ligtype in possible_Ligtypes:
                rot_vals.append([])
                possible_saves.append(False)
                for coreType in core_geo_class.liglist_geo_map_dict[ligtype]:
                    corecoordList = core_geo_class.geometry_dict[coreType]
                    # Pick first one.
                    core_cons = core_geo_class.liglist_geo_map_dict[ligtype][coreType][0]
                    tmp_ligList = []
                    for i in range(len(ligcoords)):
                        tmp_ligList.append([ligcoords[i],core_cons[i]])
                    # Test mapping on just single ligand conformation.
                    # Minval represents the loss
                    _, minval, sane, _, _, _, _ = io_lig.get_aligned_conformer(
                                ligsmiles, tmp_ligList, corecoordList, metal=metal, 
                                covrad_metal=tcovrad_metal, vdwrad_metal=tvdwrad_metal)
                    if sane:
                        rot_vals[-1].append(minval)
                        possible_saves[-1] = True
            scores = [np.mean(x) for x in rot_vals]
            min_ind = np.argsort(scores)[0]
        if possible_saves[min_ind]:
            print('Assigning lig {} to ligType {}!'.format(ligsmiles,possible_Ligtypes[min_ind]))
            return possible_Ligtypes[min_ind]
        else:
            raise ValueError('Cannot assign lig {} to any ligType!'.format(ligsmiles))


def assign_ligType_similarity(ligsmiles, ligcoords, metal, covrad_metal=None, m_diff_factor=0.5,
                              full_diff_factor=0.9):
    """assign_ligType_similarity
    Assign the ligand type based on similarity to reference ligands.

    Parameters
    ----------
    ligsmiles : str
        smiles of the lignad
    ligcoords : list(int)
        list of the coordination atoms of the ligand to the metal center
    metal : str
        metal identity
    covrad_metal : float, optional
        covalent radii of the metal, by default None
    m_diff_factor : float, optional
        fraction of the similarity of ligands dictated by metal covalent radii, by default 0.5
    full_diff_factor : float, optional
        stopping value, when similarity is close enough to consider ligand matched, by default 0.7

    Returns
    -------
    ligtype : str
        type of the ligand based on similarity to the training set ligands.
    ligmol2 : str
        ligmol2 of the similarity-matched ligand from the training set.
    """
    print('ligType not specified for {} - testing ligand placement to determine ligType!'.format(ligsmiles))
    print('Possibly slow - using similarity to training data for assignment!')
    OBmol = io_obabel.get_obmol_smiles(ligsmiles)
    rings = OBmol.GetSSSR()
    is_cp = False
    for ring in rings:
        if all(ring.IsInRing(x+1) for x in ligcoords) and (len(ligcoords) > 2) and (ring.IsAromatic()):
            is_cp = True
    ligtype = None
    ligmol2 = None
    if is_cp:
        ligtype = 'sandwich'
    elif len(ligcoords) == 1:
        ligtype = 'mono'
    elif len(ligcoords) > 9:
        ligtype = str(len(ligcoords))
    else:
        arch_path = '/'.join(architector.__file__.split('/')[0:-1])
        data_path = arch_path + '/data/angle_stats_datasource.csv'
        lig_obmol = io_obabel.get_obmol_smiles(ligsmiles)
        ligfp = io_obabel.get_fingerprint(lig_obmol)
        ligrefdf = pd.read_csv(data_path)
        ligrefdf = ligrefdf[['cn','mol2string','geotype_label']] # Downselect for only needed info
        # Add in reference ligands (search through first)
        tmp_list = list()
        for _,row in io_ptable.ligands_dict.items():
            refdict = dict()
            refdict['cn'] = len(row['coordList'])
            complexMol = io_obabel.smiles2Atoms('['+metal+']', addHydrogens=False)
            mol = io_molecule.Molecule() # Initialize molecule.
            mol.load_ase(complexMol.copy(),atom_types=[complexMol[0].symbol])
            obmollig = io_obabel.get_obmol_smiles(row['smiles'],addHydrogens=True,neutralize=False,build=False)
            bestConformer = io_obabel.convert_obmol_ase(obmollig,posits=None,set_zero=True)
            io_obabel.add_dummy_metal(obmollig,row['coordList'])
            bo_dict, atypes = io_obabel.get_OBMol_bo_dict_atom_types(obmollig)
            mol.append_ligand({'ase_atoms':bestConformer,'bo_dict':bo_dict, 
                                        'atom_types':atypes})
            refdict['mol2string'] = mol.write_mol2('cool',writestring=True)
            refdict['geotype_label'] = row['ligType']
            tmp_list.append(refdict)
        newdf = pd.DataFrame(tmp_list)
        ligrefdf = pd.concat([newdf,ligrefdf])
        ligrefdf = ligrefdf[ligrefdf.cn == len(ligcoords)]
        if covrad_metal is None:
            covrad_metal = io_ptable.rcov1[io_ptable.elements.index(metal)]
        max_diff_covrad = max(io_ptable.rcov1) - min(io_ptable.rcov1)
        min_full_diff = 0
        for _,row in ligrefdf.iterrows():
            refobmol = io_obabel.convert_mol2_obmol(row['mol2string'])
            _, anums, _ = io_obabel.get_OBMol_coords_anums_graph(refobmol, return_coords=False, get_types=False)
            syms = [io_ptable.elements[x] for x in anums]
            refmet = [x for x in syms if x in io_ptable.all_metals][0]
            ref_cov_radii = io_ptable.rcov1[io_ptable.elements.index(refmet)]
            refobmol = io_obabel.remove_obmol_metals(refobmol)
            refpf = io_obabel.get_fingerprint(refobmol)
            tanimoto_sim = refpf | ligfp # Returns tanomoto similarity between two fingerprints
            m_sim = 1 - np.abs(ref_cov_radii - covrad_metal)/max_diff_covrad
            full_diff = tanimoto_sim + m_diff_factor*m_sim
            ##Debug
            # print(io_obabel.canonicalize_smiles(ligsmiles),io_obabel.get_smiles_obmol(refobmol,canonicalize=True),full_diff)
            if isnotebook(): # Hack to handle OBmol weirdness inside jupyter. Really not sure why needed.
                print()
            if full_diff > min_full_diff:
                ligtype = row['geotype_label']
                ligmol2 = row['mol2string']
                min_full_diff = full_diff
                if min_full_diff > full_diff_factor:
                    break
    print('Assigning lig {} to ligType {}!'.format(ligsmiles,ligtype))
    return ligtype,ligmol2

    
def inparse(inputDict):
    """inparse parse all of the input!

    Parameters
    ----------
    inputDict : dict
        input Dictionary

    Returns
    -------
    newinpDict : dict
        new input Dictionary with full information after parsing.

    Raises
    ------
    ValueErrors -> Where things are unknown to architector.
    """
    newinpDict = inputDict.copy()
    u_id = str(uuid.uuid4())
    # Adding logging
    # logging.config.fileConfig('/path/to/logging.conf')
    if (('core' in inputDict) and (('ligands' in inputDict) or ('ligandList' in inputDict))) or ('mol2string' in inputDict):
        if (('core' in inputDict) and (('ligands' in inputDict) or ('ligandList' in inputDict))):
            coreDict = newinpDict['core']
            #######################################################################
            ############### Process metal/core input first. #######################
            #######################################################################
            if 'metal' in coreDict:
                coreDict['smiles'] = '['+coreDict['metal']+']'
            elif 'smiles' in coreDict:
                if ('[' not in coreDict['smiles']) or (']' not in coreDict['smiles']):
                    print('Warning: attempting to add parenthesis to this smiles string: ', coreDict['smiles'])
                    coreDict['smiles'] = '['+coreDict['smiles'] + ']'
            else:
                print('No metal/core passed - defaulting to Fe.')
                coreDict['smiles'] = '[Fe]'

            metal = newinpDict['core']['smiles'].strip('[').strip(']')

            # Convert actinide to lanthanide if necessary
            tmetal, is_actinide = io_ptable.convert_actinides_lanthanides(metal)

            newinpDict['core']['smiles'] = '['+tmetal+']'
            newinpDict['core']['metal'] = tmetal
            if isinstance(newinpDict['parameters'],dict):
                newinpDict['parameters']['is_actinide'] = is_actinide
                newinpDict['parameters']['original_metal'] = metal
            else:
                newinpDict['parameters'] = dict()
                newinpDict['parameters']['is_actinide'] = is_actinide
                newinpDict['parameters']['original_metal'] = metal

            skip = False
            coreTypes = []
            if 'coordList' in coreDict:
                if isinstance(coreDict['coordList'],list):
                    print('Adding user core geometry. Locking coreCN to match user core geometry.')
                    ## Process core geometry -> Load geometries
                    core_geo_class = io_core.Geometries(usercore=coreDict['coordList'])
                    coreTypes = ['user_core']
                    skip = True
                elif (coreDict['coordList'] is None) or isinstance(coreDict['coordList'],bool):
                    skip = False
                else:
                    raise ValueError('Unrecognized type passed to inputDict["core"]["coordList"] - need list or None/bool.')
            if ('coreType' in coreDict) and (not skip):
                if isinstance(coreDict['coreType'],list):
                    core_geo_class = io_core.Geometries()
                    for x in coreDict['coreType']:
                        if x in core_geo_class.geometry_dict:
                            coreTypes.append(x)
                            skip = True
                        else:
                            print('{} not in known coreTypes. Skipping.'.format(x))
                            print('Known coreTypes: ', core_geo_class.geometry_dict.keys())
                elif (coreDict['coreType'],str):
                    core_geo_class = io_core.Geometries()
                    if coreDict['coreType'] in core_geo_class.geometry_dict.keys():
                        coreTypes.append(coreDict['coreType'])
                        skip = True
                    else:
                        print('{} not in known coreTypes. Skipping.'.format(coreDict['coreType']))
                        print('Known coreTypes: ', core_geo_class.geometry_dict.keys())
                elif (coreDict['coreType'] is None) or isinstance(coreDict['coreType'],bool):
                    skip = False
                else:
                    raise ValueError('Unrecognized type passed to inputDict["core"]["coreType"] - need list/str/None/bool.')
            if ('coreCN' in coreDict) and (not skip):
                if isinstance(coreDict['coreCN'],list):
                    core_geo_class = io_core.Geometries()
                    for x in coreDict['coreCN']:
                        if x in core_geo_class.cn_geo_dict.keys():
                            coreTypes += core_geo_class.cn_geo_dict[x]
                            skip = True
                        else:
                            print('{} not a valid coreCN.'.format(x))
                            print('Valid coreCNs are: ', core_geo_class.cn_geo_dict.keys())
                elif isinstance(coreDict['coreCN'],(int,float)):
                    core_geo_class = io_core.Geometries()
                    cn = int(coreDict['coreCN'])
                    if cn in core_geo_class.cn_geo_dict.keys():
                        coreTypes += core_geo_class.cn_geo_dict[cn]
                        skip = True
                    else:
                        print('{} not a valid coreCN.'.format(cn))
                        print('Valid coreCNs are: ', core_geo_class.cn_geo_dict.keys())
                elif (coreDict['coreType'] is None) or isinstance(coreDict['coreType'],bool):
                    core_geo_class = io_core.Geometries()
                    print("At this point not sure what is up. Defaulting to coreCN=6.")
                    coreTypes += core_geo_class.cn_geo_dict[6]
                else:
                    raise ValueError('Unrecognized type passed to inputDict["core"]["coreCN"] - need list/int/float/None/bool.')
            elif (not skip): # Use default list of CNs to generate complexes.
                core_geo_class = io_core.Geometries()
                for x in io_ptable.metal_CN_dict[metal]:
                    coreTypes += core_geo_class.cn_geo_dict[x]
            # Catch cases where no coretypes known.
            if len(coreTypes) == 0:
                raise ValueError('No coreTypes defined!!!!!!')

            #######################################################################
            ############### Next map ligands to metal/core. #######################
            #######################################################################
            covrad_metal = None
            vdwrad_metal = None
            if ('parameters' in newinpDict): # Load input parameters as changes to the defaults.
                if isinstance(newinpDict['parameters'],dict):
                    if 'covrad_metal' in newinpDict['parameters']:
                        covrad_metal = newinpDict['parameters']['covrad_metal']
                    if 'vdwrad_metal' in newinpDict['parameters']:
                        vdwrad_metal = newinpDict['parameters']['vdwrad_metal']

            core_geo_class.get_lig_ref_inds_dict(metal, coreTypes, rcovmetal=covrad_metal) # Calculate core references -> rescaled

            # Will need to add new ligand reference when capability added.
            if 'ligands' in newinpDict:
                # Check Ligand list
                ligList = newinpDict['ligands']
            elif 'ligandList' in newinpDict: # Reassign
                ligList = newinpDict['ligandList']
                newinpDict['ligands'] = ligList
            else:
                raise ValueError('Need a inputDict["ligands"] list passed')
            newliglist = []

            if isinstance(ligList, list):
                for ligDict in ligList:
                    newdict = dict()
                    if isinstance(ligDict,dict):
                        if ('smiles' in ligDict) and ('coordList' in ligDict) and ('ligType' in ligDict):
                            newdict.update({'smiles':ligDict['smiles']})
                            if (ligDict['smiles'] == '[H-]')or (ligDict['smiles'] == '[O-2]'): # Handle hydrides
                                newinpDict['parameters']['relax'] = False
                                newinpDict['parameters']['metal_spin'] = 0 # Set to low_spin
                            if (ligDict['ligType'] in core_geo_class.liglist_geo_map_dict.keys()) or (ligDict['ligType'] == 'mono'):
                                newdict.update({'ligType':ligDict['ligType']})
                            else:
                                print('Error: {} ligand type not recognized!'.format(ligDict['ligType']))
                                print('Valid ligand types: ', core_geo_class.liglist_geo_map_dict.keys(), ' + mono')
                                raise ValueError
                            if isinstance(ligDict['coordList'][0],(int,float)):
                                newdict.update({'coordList':[int(x) for x in ligDict['coordList']]})
                            elif isinstance(ligDict['coordList'][0],list):
                                newdict.update({'coordList':ligDict['coordList']})
                            else:
                                raise ValueError('Error: {} ligand coordList not recognized!'.format(ligDict['coordList']))
                        elif ('smiles' in ligDict) and ('coordList' in ligDict):
                            # Need specific ligand mapping
                            newdict.update({'smiles':ligDict['smiles']})
                            if (ligDict['smiles'] == '[H-]') or (ligDict['smiles'] == '[O-2]'): # Handle hydrides
                                newinpDict['parameters']['relax'] = False
                                newinpDict['parameters']['metal_spin'] = 0 # Set to low-spin for oxos
                            if isinstance(ligDict['coordList'][0],list):
                                newdict.update({'coordList':ligDict['coordList']})
                                # Assign ligType arbitrarily -> will be ignored by io_symmetry if list of lists
                                newdict.update({'ligType':'mono'})
                            elif len(ligDict['coordList']) == 1: # Monodentate
                                newdict.update({'coordList':ligDict['coordList']})
                                newdict.update({'ligType':'mono'})
                            elif len(ligDict['coordList']) > 1: # Handle other cases with bruteforce assignemnt of ligandType!
                                newdict.update({'coordList':ligDict['coordList']})
                                bruteforce = True
                                for tdict in newliglist: # Check other ligands if ligType already assigned.
                                    if (tdict['smiles'] == newdict['smiles']) and (sorted(tdict['coordList']) == sorted(newdict['coordList'])):
                                        newdict.update({'ligType':tdict['ligType']})
                                        bruteforce = False
                                        break
                                if bruteforce:
                                    ###### Currently have the bruteforce ligand type assignemnt and similarity routine for 2D to 3D.
                                    if 'lig_assignment' in newinpDict['parameters']:
                                        if newinpDict['parameters']['lig_assignment'] == 'bruteforce':
                                            tmp_ligType = assign_ligType_bruteforce(core_geo_class, newdict['smiles'], newdict['coordList'], 
                                                                                    tmetal, covrad_metal=covrad_metal, vdwrad_metal=vdwrad_metal)
                                        elif newinpDict['parameters']['lig_assignment'] == 'similarity':
                                            tmp_ligType,_ = assign_ligType_similarity(newdict['smiles'], newdict['coordList'], 
                                                                                    tmetal, covrad_metal=covrad_metal)
                                    else:
                                        tmp_ligType = assign_ligType_bruteforce(core_geo_class, newdict['smiles'], newdict['coordList'], 
                                                tmetal, covrad_metal=covrad_metal, vdwrad_metal=vdwrad_metal)
                                    newdict.update({'ligType':tmp_ligType})
                            else:
                                raise ValueError('Error: {} ligand coordList not recognized!'.format(ligDict['coordList']))
                        elif ('smiles' in ligDict): # Handle simple ligands without needed specs.
                            if ligDict['smiles'] == '[H-]': # Handle hydrides
                                newdict.update({'smiles':ligDict['smiles']})
                                newdict.update({'coordList':[0]})
                                newdict.update({'ligType':'mono'})
                                newinpDict['parameters']['relax'] = False
                            elif ligDict['smiles'] == 'O':
                                newdict.update({'smiles':ligDict['smiles']})
                                newdict.update({'coordList':[0]})
                                newdict.update({'ligType':'mono'})
                            elif ligDict['smiles'] == '[OH-]':
                                newdict.update({'smiles':ligDict['smiles']})
                                newdict.update({'coordList':[0]})
                                newdict.update({'ligType':'mono'})
                            elif ligDict['smiles'] == '[O-2]':
                                newdict.update({'smiles':ligDict['smiles']})
                                newdict.update({'coordList':[0]})
                                newdict.update({'ligType':'mono'})
                                newinpDict['parameters']['relax'] = False
                            else:
                                raise ValueError('This smiles is not recognized - need coordList/ligType assignment {}!'.format(ligDict['smiles']))
                        elif ('name' in ligDict): # Read from io_ptable reference ligands!
                            if ligDict['name'].lower() in io_ptable.ligands_dict:
                                newdict = copy.deepcopy(io_ptable.ligands_dict[ligDict['name'].lower()])
                            else:
                                raise ValueError("""This ligand's {} name is not recognized. See architector.io_ptable for reference ligands.
                                 Note that this should be case-insensitive. """.format(ligDict['name']))
                        else:
                            raise ValueError('Need at least "smiles" and "coordList" for each ligand or "name"!')
                    elif isinstance(ligDict,str): # Add ligand from the reference 
                        if ligDict.lower() in io_ptable.ligands_dict:
                            newdict = copy.deepcopy(io_ptable.ligands_dict[ligDict.lower()])
                        else:
                            raise ValueError("""This ligand's {} name is not recognized. See architector.io_ptable for reference ligands.
                                Note that this should be case-insensitive. """.format(ligDict))
                    else:
                        raise ValueError('Unrecognized ligands definition!')
                    if isinstance(ligDict,dict):
                        if ('functionalizations' in ligDict):
                            tmpOBmol = io_obabel.get_obmol_smiles(newdict['smiles'],functionalizations=ligDict['functionalizations'])
                            newsmi = io_obabel.get_smiles_obmol(tmpOBmol)
                            newdict.update({'smiles':newsmi})
                    newliglist.append(newdict)
            else:
                raise ValueError("inputDict['ligands'] must be a list of dictionaries currently.")
        ####################################
        ####### Mol2 string parsing ######## -> Most useful for 2D to 3D generation
        ####################################
        elif ('mol2string' in inputDict):
            # Read in charge/spin from the mol2 string in case assigned.
            mol = io_molecule.convert_io_molecule(inputDict['mol2string'])
            if (mol.charge is not None):
                if 'parameters' not in newinpDict:
                    newinpDict['parameters'] = dict()
                if isinstance(newinpDict['parameters'],dict):
                    newinpDict['parameters']['full_charge'] = mol.charge
                else:
                    newinpDict['parameters'] = dict()
                    newinpDict['parameters']['full_charge'] = mol.charge
            if (mol.uhf is not None):
                if 'parameters' not in newinpDict:
                    newinpDict['parameters'] = dict()
                if isinstance(newinpDict['parameters'],dict):
                    newinpDict['parameters']['full_spin'] = mol.uhf
                else:
                    newinpDict['parameters'] = dict()
                    newinpDict['parameters']['full_spin'] = mol.uhf
            if (mol.xtb_uhf is not None): # Currently default to XTB spin -> xtb backend makes more sense
                if 'parameters' not in newinpDict:
                    newinpDict['parameters'] = dict()
                if isinstance(newinpDict['parameters'],dict):
                    newinpDict['parameters']['full_spin'] = mol.xtb_uhf
                else:
                    newinpDict['parameters'] = dict()
                    newinpDict['parameters']['full_spin'] = mol.xtb_uhf
            metal_inds = [i for i,x in enumerate(mol.ase_atoms.get_chemical_symbols()) if x in io_ptable.all_metals]
            if len(metal_inds) == 1: # Only one metal - perform ligand breakdown to find all ligands.
                ##### Core Preprocessing ####
                met_ind = metal_inds[0]
                metal = mol.ase_atoms.get_chemical_symbols()[met_ind]
                # Convert actinide to lanthanide if necessary
                tmetal, is_actinide = io_ptable.convert_actinides_lanthanides(metal)
                newinpDict['core'] = dict()
                newinpDict['core']['smiles'] = '['+tmetal+']'
                newinpDict['core']['metal'] = tmetal
                if isinstance(newinpDict['parameters'],dict):
                    newinpDict['parameters']['is_actinide'] = is_actinide
                    newinpDict['parameters']['original_metal'] = metal
                else:
                    newinpDict['parameters'] = dict()
                    newinpDict['parameters']['is_actinide'] = is_actinide
                    newinpDict['parameters']['original_metal'] = metal
                newinpDict['core']['coreCN'] = len(np.nonzero(np.ravel(mol.graph[met_ind]))[0])
                core_geo_class = io_core.Geometries()
                cn = int(newinpDict['core']['coreCN'])
                coreTypes = list()
                if cn in core_geo_class.cn_geo_dict.keys():
                    coreTypes += core_geo_class.cn_geo_dict[cn]
                    skip = True
                else:
                    print('{} not a valid coreCN.'.format(cn))
                    print('Valid coreCNs are: ', core_geo_class.cn_geo_dict.keys())
                ############### Next map ligands to metal/core. #######################
                covrad_metal = None
                vdwrad_metal = None
                if ('parameters' in newinpDict): # Load input parameters as changes to the defaults.
                    if isinstance(newinpDict['parameters'],dict):
                        if 'covrad_metal' in newinpDict['parameters']:
                            covrad_metal = newinpDict['parameters']['covrad_metal']
                        if 'vdwrad_metal' in newinpDict['parameters']:
                            vdwrad_metal = newinpDict['parameters']['vdwrad_metal']

                core_geo_class.get_lig_ref_inds_dict(metal, coreTypes, rcovmetal=covrad_metal) # Calculate core references -> rescaled
                lig_smiles,coord_atoms = io_obabel.obmol_lig_split(inputDict['mol2string'])
                newliglist = list()
                for i,lig_smiles in enumerate(lig_smiles):
                    tligdict = {'smiles':lig_smiles,'coordList':coord_atoms[i]}
                    if (tligdict['smiles'] == '[H-]') or (tligdict['smiles'] == '[O-2]'): # Handle hydrides
                        newinpDict['parameters']['relax'] = False
                        newinpDict['parameters']['metal_spin'] = 0 # Set to low-spin for oxos
                        newinpDict['parameters']['full_spin'] = None
                        tligdict.update({'ligType':'mono'})
                    elif len(tligdict['coordList']) == 1: # Monodentate
                        tligdict.update({'ligType':'mono'})
                    elif len(tligdict['coordList']) > 1: # Handle other cases with bruteforce assignemnt of ligandType!
                        bruteforce = True
                        for tdict in newliglist: # Check other ligands if ligType already assigned.
                            if (tdict['smiles'] == tligdict['smiles']) and (sorted(tdict['coordList']) == sorted(tligdict['coordList'])):
                                tligdict.update({'ligType':tdict['ligType']})
                                bruteforce = False
                                break
                        if bruteforce:
                            ###### Currently default to similariy ligand type assignemnt routine for 2D to 3D.
                            if 'lig_assignment' in newinpDict['parameters']:
                                if newinpDict['parameters']['lig_assignment'] == 'bruteforce':
                                    tmp_ligType = assign_ligType_bruteforce(core_geo_class, tligdict['smiles'], tligdict['coordList'], 
                                        tmetal, covrad_metal=covrad_metal, vdwrad_metal=vdwrad_metal)
                                elif newinpDict['parameters']['lig_assignment'] == 'similarity':
                                    tmp_ligType,_ = assign_ligType_similarity(tligdict['smiles'], tligdict['coordList'], 
                                        tmetal, covrad_metal=covrad_metal,full_diff_factor=0.8)
                            else:
                                tmp_ligType,_ = assign_ligType_similarity(tligdict['smiles'], tligdict['coordList'], 
                                                                        tmetal, covrad_metal=covrad_metal,full_diff_factor=0.8)
                            tligdict.update({'ligType':tmp_ligType})
                    newliglist.append(tligdict)
            else:
                raise ValueError('Either less than 1 or more than 1 metal in this mol2string passed.')

        newinpDict['ligands'] = newliglist

        coreTypes_run = []
        for coreType in coreTypes:
            good = True
            for lig in newinpDict['ligands']: # Check that all ligands can map to this geometry
                if lig['ligType'] == 'mono':
                    good = True
                elif (coreType not in core_geo_class.liglist_geo_map_dict[lig['ligType']].keys()):
                    good=False
            if good:
                coreTypes_run.append(coreType)

        if len(coreTypes_run) == 0:
            print('Warning: No structures generated! No Coretype/ligand type match.')
            print('Attempting ligand type assignment')
            new_liglist = []
            for tligdict in newinpDict['ligands']:
                tmp_ligType = assign_ligType_bruteforce(core_geo_class, tligdict['smiles'], tligdict['coordList'], 
                                        tmetal, covrad_metal=covrad_metal, vdwrad_metal=vdwrad_metal)
                tligdict.update({'ligType': tmp_ligType})
                new_liglist.append(tligdict)
            newinpDict['ligands'] = newliglist
            coreTypes_run = []
            for coreType in coreTypes:
                good = True
                for lig in newinpDict['ligands']: # Check that all ligands can map to this geometry
                    if lig['ligType'] == 'mono':
                        good = True
                    elif (coreType not in core_geo_class.liglist_geo_map_dict[lig['ligType']].keys()):
                        good=False
                if good:
                    coreTypes_run.append(coreType)
        
        # Reset coreTypes             
        coreTypes = coreTypes_run

        if len(coreTypes) == 0:
            print("Warning: No structures generated! Still couldn't map the ligands to the core")

        
        # Store saved info
        newinpDict['coreTypes'] = coreTypes
        newinpDict['core_geo_class'] = core_geo_class

        #######################################################################
        ################## Finally - set parameters.  #########################
        #######################################################################

        default_parameters = {
            "n_conformers":1, # Number of metal-core symmetries at each core to save / relax
            "return_only_1":False, # Only return single relaxed conformer (do not test multiple conformations)
            "n_symmetries":10, # Total metal-center symmetrys to build, NSymmetries should be >= n_conformers
            # 'n_lig_combos':1, # Number of randomized ligand conformations to run/return for each conformer -> possibly add back
            "crest_sampling":False, # Perform CREST sampling on lowest-energy conformer?
            "relax": True, # Perform xTB geomtetry relaxation of assembled complexes
            "debug": False, # Print out additional info for debugging purposes.
            "save_init_geos": False, # Save initial geometries before relaxations with xTB.
            "seed":None, # If a seed is passed (int/float) use it to initialize np.random.seed for reproducability.
            # If you want to replicate whole workflows - set np.random.seed() at the beginning of your workflow.
            ### OPENBABEL STILL HAS RANDOMNESS
            "save_trajectories": False, # Save full relaxation trajectories from xTB to the ase db.
            "return_timings":True, # Return all the timings.
            "return_full_complex_class":False, # Return the complex class containing all ligand geometry and core information.
            "uid":u_id, # Unique ID (generated by default, but can be assigned)

            # Dump all possible intermediate xtb calcs to separate database
            "dump_ase_atoms": False, # or True
            "ase_atoms_db_name": 'architector_{}_ase_db_'.format(newinpDict['core']['metal']) + u_id + '.json',
            "temp_prefix":"/tmp/", # Default here - for MPI running on HPC suggested /scratch/$USER/
            
            # Ligand parameters
            # Ligand to finish filling out coordination environment if underspecified.
            "fill_ligand": "water", 
            # Secondary fill ligand will be a monodentate ligand to fill out coordination environment
            # in case the fill_ligand and specified ligands list cannot fully map to the coordination environment.
            "secondary_fill_ligand": "water",
            # or integer index in reference to the ligand list!!
            "force_trans_oxos":False, # Force trans configurations for oxos (Useful for actinyls)
            "lig_assignment":'bruteforce', # or "similarity" How to automatically assign ligand types.

            # Cutoff parameters
            "assemble_sanity_checks":True, # Turn on/off assembly sanity checks.
            "assemble_graph_sanity_cutoff":1.8,
            # Graph Sanity cutoff for imposed molecular graph represents the maximum elongation of bonds
            # rcov1*full_graph_sanity_cutoff is the maximum value for the bond lengths.
            "assemble_smallest_dist_cutoff":0.3,
            # Smallest dist cutoff screens if any bonds are less than smallest_dist_cutoff*sum of cov radii
            # Will not be evaluated by XTB if they are lower.
            "assemble_min_dist_cutoff":4,
            # Smallest min dist cutoff screens if any atoms are at minimum min_dist_cutoff*sum of cov radii
            # away from ANY other atom (indicating blown-up structure) 
            # - will not be evaluated by XTB if they are lower.
            "full_sanity_checks":True, # Turn on/off final sanity checks.
            "full_graph_sanity_cutoff":1.7,
            # full_graph_sanity_cutoff can be tightened to weed out distorted geometries (e.g. 1.5 for non-group1-metals) 
            "full_smallest_dist_cutoff":0.55,
            "full_min_dist_cutoff":3.5,

            # Electronic parameters
            "metal_ox": None, # Oxidation State
            "metal_spin": None, # Spin State
            "test_alternate_metal_spin": False, # Test a second spin state for the metal?
            "alternate_metal_spin": None, # Secondary spin state to check. 
                
            # Method parameters.
            "xtb_solvent": 'none', # Add any named XTB solvent!
            "xtb_accuracy":1.0, # Numerical Accuracy for XTB calculations
            "xtb_electronic_temperature":300, # In K -> fermi smearing - increase for convergence on harder systems
            "xtb_max_iterations":250, # Max iterations for xtb SCF.
            "full_spin": None, # Assign spin to the full complex (overrides metal_spin)
            "full_charge": None, # Assign charge to the complex (overrides ligand charges and metal_ox)!
            "full_method":"GFN2-xTB", # Which method to use for final cleaning/evaulating conformers.
            "assemble_method":"GFN2-xTB", # Which method to use for assembling conformers. 
            "fmax":0.1, # eV/Angstrom - max force for relaxation.
            "maxsteps":1000, # Steps involved in relaxation
            "force_generation":False, # Whether to force the construction to proceed without xtb energies - defaults to UFF
            # In cases of XTB outright failure.
            # For very large speedup - use GFN-FF, though this is much less stable (especially for Lanthanides)
            # Or UFF

            "skip_duplicate_tests":False, # Skip the duplicate tests (return all generated/relaxed configurations)
            
            # Covalent radii and vdw radii of the metal if deviations requested.
            "vdwrad_metal":vdwrad_metal,
            "covrad_metal":covrad_metal,
            "scaled_radii_factor":None, # Bookeeping if scaled vdwrad/covrad passed.
        } 

        outparams = dict()

        # Default to GFN-FF for assemble methods for heavy metals - GFN2-XTB assembly takes A LOT longer on these systems.
        # if newinpDict['core']['metal'] in io_ptable.heavy_metals:
        #     default_parameters['assemble_method'] = 'GFN-FF'

        outparams.update(default_parameters) 
        # If acinide default to forcing trans oxos if 2 present (actinyls)
        # Can still be turned off if desired.
        if newinpDict['parameters']['is_actinide']:
            count = 0 
            for lig in newinpDict['ligands']:
                if lig['smiles'] == '[O-2]':
                    count += 1
            if count == 2:
                newinpDict['parameters']['force_trans_oxos'] = True

        if ('parameters' in newinpDict): # Load input parameters as changes to the defaults.
            if isinstance(newinpDict['parameters'],dict):
                outparams.update(newinpDict['parameters'])

        if outparams['n_conformers'] < 1:
            print('Defaulting to 1 conformer')
            outparams['n_conformers'] = 1
        elif outparams['n_conformers'] > 50:
            print('Way too many conformers requested. Defaulting to 50')
            outparams['n_conformers'] = 50
            outparams['n_symmetries'] = 50

        if outparams['force_generation']:
            # Graph Sanity cutoff for imposed molecular graph represents the maximum elongation of bonds
            # rcov1*full_graph_sanity_cutoff is the maximum value for the bond lengths.
            outparams["assemble_smallest_dist_cutoff"] = 0.7 # Force this to be tighter if no optimization done.
            outparams["full_smallest_dist_cutoff"] = 0.7 # Force this to be tighter if no optimization done.

        # Make looser final cutoff allowances for graph distances for alkali and alkali earth metals
        # Tend to form more ionic bonds that may get elongated.
        if newinpDict['core']['metal'] in io_ptable.alkali_and_alkaline_earth:
            outparams['full_graph_sanity_cutoff'] = 1.8

        # Push out full graph sanity for cp rings.
        if any([True for x in newinpDict['ligands'] if (x['ligType'] == 'sandwich')]):
            outparams['full_graph_sanity_cutoff'] = 2.0

        # Convert fill ligands information into correct graph for default ligands
        if isinstance(outparams['fill_ligand'],int):
            outparams['fill_ligand'] = newinpDict['ligands'][outparams['fill_ligand']]
        elif isinstance(outparams['fill_ligand'],str):
            if outparams['fill_ligand'] in io_ptable.ligands_dict:
                outparams['fill_ligand'] = io_ptable.ligands_dict[outparams['fill_ligand']]
            else: 
                raise ValueError('Unrecognized ligand name: {}'.format(outparams['fill_ligand']))
        
        if isinstance(outparams['secondary_fill_ligand'],str):
            if outparams['secondary_fill_ligand'] in io_ptable.ligands_dict:
                outparams['secondary_fill_ligand'] = io_ptable.ligands_dict[outparams['secondary_fill_ligand']]
            else: 
                raise ValueError('Unrecognized ligand name: {}'.format(outparams['secondary_fill_ligand']))
        
        if outparams['n_conformers'] > outparams['n_symmetries']:
            print('Shifting symmetries to match conformers.')
            outparams['n_symmetries'] = outparams['n_conformers']

        if outparams['metal_ox'] is None:
            outparams['metal_ox'] = io_ptable.metal_charge_dict[metal]
        if outparams['metal_spin'] is None:
            if outparams['metal_ox'] != io_ptable.metal_charge_dict[metal]:
                # Calculate from mendeleev reference - Generally aufbau.
                outparams['metal_spin'] = mendeleev.__dict__[newinpDict['core']['metal']].ec.ionize(outparams['metal_ox']).unpaired_electrons()
            else: # Otherwise use refdict.
                outparams['metal_spin'] = io_ptable.metal_spin_dict[metal]

        if outparams['alternate_metal_spin'] is None:
            if metal in io_ptable.second_choice_metal_spin_dict:
                outparams['alternate_metal_spin'] = io_ptable.second_choice_metal_spin_dict[metal]

        # Connect to ase database
        outparams['ase_db_tmp_name'] = os.path.join(outparams['temp_prefix'],outparams['ase_atoms_db_name'])
        if outparams['dump_ase_atoms']:
            db = connect(outparams['ase_db_tmp_name'], serial=True)
            outparams['ase_db'] = db
        elif outparams['save_trajectories']:
            outparams['dump_ase_atoms'] = True
            db = connect(outparams['ase_db_tmp_name'], serial=True)
            outparams['ase_db'] = db
        else:
            outparams['ase_db'] = None

        # Screen out non-trans geos if requested
        if outparams['force_trans_oxos']:
            coreTypes = [x for x in newinpDict['coreTypes'] if x in core_geo_class.trans_geos]
            newinpDict['coreTypes'] = coreTypes

        # # Load logger
        # if outparams['logg']
        newinpDict['parameters'] = outparams

        # Initialize seed.
        if isinstance(newinpDict['parameters']['seed'],(int,float,np.float64,np.int64)):
            np.random.seed(int(newinpDict['parameters']['seed']))
    
        return newinpDict
    else:
        raise ValueError('Input structure not recognized.')

def test_ligType_sandwich(ligsmiles, ligcoords):
    """assign_ligType_bruteforce
    Assign the ligtype to sandwich if it is

    Parameters
    ----------
    ligsmiles : str
        ligand smiles
    ligcoords : list
        1D ligand coordinating atom list

    Returns
    -------
    ligType : str
        "Sandiwch" or "none"

    """
    ## First check for cp rings -> all indices in shared ring.
    OBmol = io_obabel.get_obmol_smiles(ligsmiles)
    rings = OBmol.GetSSSR()
    is_cp = False
    for ring in rings:
        if all(ring.IsInRing(x+1) for x in ligcoords) and (len(ligcoords) > 2) and (ring.IsAromatic()):
            is_cp = True
    if is_cp:
        return 'sandwich'
    else:
        return 'none'

def inparse_2D(inputDict):
    """Parsing for 2D molecules"""
    newinpDict = inputDict.copy()
    if ('core' in inputDict) and (('ligands' in inputDict) or ('ligandList' in inputDict)):
        coreDict = newinpDict['core']
        #######################################################################
        ############### Process metal/core input first. #######################
        #######################################################################
        if 'metal' in coreDict:
            coreDict['smiles'] = '['+coreDict['metal']+']'
        elif 'smiles' in coreDict:
            if ('[' not in coreDict['smiles']) or (']' not in coreDict['smiles']):
                print('Warning: attempting to add parenthesis to this smiles string: ', coreDict['smiles'])
                coreDict['smiles'] = '['+coreDict['smiles'] + ']'
        else:
            print('No metal/core passed - defaulting to Fe.')
            coreDict['smiles'] = '[Fe]'

        metal = newinpDict['core']['smiles'].strip('[').strip(']')

        skip = False
        coreTypes = []
        if 'coordList' in coreDict:
            if isinstance(coreDict['coordList'],list):
                print('Adding user core geometry. Locking coreCN to match user core geometry.')
                ## Process core geometry -> Load geometries
                core_geo_class = io_core.Geometries(usercore=coreDict['coordList'])
                coreTypes += core_geo_class.cn_geo_dict[len(coreDict['coordList'])]
                skip = True
            elif (coreDict['coordList'] is None) or isinstance(coreDict['coordList'],bool):
                skip = False
            else:
                raise ValueError('Unrecognized type passed to inputDict["core"]["coordList"] - need list or None/bool.')
        if ('coreType' in coreDict) and (not skip):
            if isinstance(coreDict['coreType'],list):
                core_geo_class = io_core.Geometries()
                for x in coreDict['coreType']:
                    if x in core_geo_class.geometry_dict:
                        coreTypes.append(x)
                        skip = True
                    else:
                        print('{} not in known coreTypes. Skipping.'.format(x))
                        print('Known coreTypes: ', core_geo_class.geometry_dict.keys())
            elif (coreDict['coreType'],str):
                core_geo_class = io_core.Geometries()
                if coreDict['coreType'] in core_geo_class.geometry_dict.keys():
                    coreTypes.append(coreDict['coreType'])
                    skip = True
                else:
                    print('{} not in known coreTypes. Skipping.'.format(coreDict['coreType']))
                    print('Known coreTypes: ', core_geo_class.geometry_dict.keys())
            elif (coreDict['coreType'] is None) or isinstance(coreDict['coreType'],bool):
                skip = False
            else:
                raise ValueError('Unrecognized type passed to inputDict["core"]["coreType"] - need list/str/None/bool.')
        if ('coreCN' in coreDict) and (not skip):
            if isinstance(coreDict['coreCN'],list):
                core_geo_class = io_core.Geometries()
                for x in coreDict['coreCN']:
                    if x in core_geo_class.cn_geo_dict.keys():
                        coreTypes += core_geo_class[x]
                        skip = True
                    else:
                        print('{} not a valid coreCN.'.format(x))
                        print('Valid coreCNs are: ', core_geo_class.cn_geo_dict.keys())
            elif isinstance(coreDict['coreCN'],(int,float)):
                core_geo_class = io_core.Geometries()
                cn = int(coreDict['coreCN'])
                if cn in core_geo_class.cn_geo_dict.keys():
                    coreTypes += core_geo_class.cn_geo_dict[cn]
                    skip = True
                else:
                    print('{} not a valid coreCN.'.format(cn))
                    print('Valid coreCNs are: ', core_geo_class.cn_geo_dict.keys())
            elif (coreDict['coreType'] is None) or isinstance(coreDict['coreType'],bool):
                core_geo_class = io_core.Geometries()
                print("At this point not sure what is up. Defaulting to coreCN=6.")
                coreTypes += core_geo_class.cn_geo_dict[6]
            else:
                raise ValueError('Unrecognized type passed to inputDict["core"]["coreCN"] - need list/int/float/None/bool.')
        elif (not skip): # Use default list of CNs to generate complexes.
            core_geo_class = io_core.Geometries()
            for x in io_ptable.metal_CN_dict[metal]:
                coreTypes += core_geo_class.cn_geo_dict[x]

        # Catch cases where no coretypes known.
        if len(coreTypes) == 0:
            raise ValueError('No coreTypes defined!!!!!!')

        #######################################################################
        ############### Next map ligands to metal/core. #######################
        #######################################################################
        covrad_metal = None
        vdwrad_metal = None
        if ('parameters' in newinpDict): # Load input parameters as changes to the defaults.
            if isinstance(newinpDict['parameters'],dict):
                if 'covrad_metal' in newinpDict['parameters']:
                    covrad_metal = newinpDict['parameters']['covrad_metal']
                if 'vdwrad_metal' in newinpDict['parameters']:
                    vdwrad_metal = newinpDict['parameters']['vdwrad_metal']

        # core_geo_class.get_lig_ref_inds_dict(metal, coreTypes, rcovmetal=covrad_metal) # Calculate core references -> rescaled

        # Will need to add new ligand reference when capability added.
        if 'ligands' in newinpDict:
            # Check Ligand list
            ligList = newinpDict['ligands']
        elif 'ligandList' in newinpDict: # Reassign
            ligList = newinpDict['ligandList']
            newinpDict['ligands'] = ligList
        else:
            raise ValueError('Need a inputDict["ligands"] list passed')
        newliglist = []

        if isinstance(ligList, list):
            for ligDict in ligList:
                newdict = dict()
                if isinstance(ligDict,dict):
                    if ('smiles' in ligDict) and ('coordList' in ligDict) and ('ligType' in ligDict):
                        newdict.update({'smiles':ligDict['smiles']})
                        if (ligDict['smiles'] == '[H-]')or (ligDict['smiles'] == '[O-2]'): # Handle hydrides
                            newinpDict['parameters']['relax'] = False
                            newinpDict['parameters']['metal_spin'] = 0 # Set to low_spin
                        if (ligDict['ligType'] == 'mono') or (isinstance(ligDict['ligType'],str)):
                            newdict.update({'ligType':ligDict['ligType']})
                        else:
                            print('Error: {} ligand type not recognized!'.format(ligDict['ligType']))
                            # print('Valid ligand types: ', core_geo_class.liglist_geo_map_dict.keys(), ' + mono')
                            raise ValueError
                        if isinstance(ligDict['coordList'][0],(int,float)):
                            newdict.update({'coordList':[int(x) for x in ligDict['coordList']]})
                        elif isinstance(ligDict['coordList'][0],list):
                            newdict.update({'coordList':ligDict['coordList']})
                        else:
                            raise ValueError('Error: {} ligand coordList not recognized!'.format(ligDict['coordList']))
                    elif ('smiles' in ligDict) and ('coordList' in ligDict):
                        # Need specific ligand mapping
                        newdict.update({'smiles':ligDict['smiles']})
                        if (ligDict['smiles'] == '[H-]') or (ligDict['smiles'] == '[O-2]'): # Handle hydrides
                            newinpDict['parameters']['relax'] = False
                            newinpDict['parameters']['metal_spin'] = 0 # Set to low-spin for oxos
                        if isinstance(ligDict['coordList'][0],list):
                            newdict.update({'coordList':ligDict['coordList']})
                            # Assign ligType arbitrarily -> will be ignored by io_symmetry if list of lists
                            newdict.update({'ligType':'mono'})
                        elif len(ligDict['coordList']) == 1: # Monodentate
                            newdict.update({'coordList':ligDict['coordList']})
                            newdict.update({'ligType':'mono'})
                        elif len(ligDict['coordList']) > 1: # Handle other cases with bruteforce assignemnt of ligandType!
                            tmp_ligType = test_ligType_sandwich(ligDict['smiles'], ligDict['coordList'])
                            newdict.update({'coordList':ligDict['coordList']})
                            newdict.update({'ligType':tmp_ligType}) # Doesn't matter much for 2D except sandwich
                        else:
                            raise ValueError('Error: {} ligand coordList not recognized!'.format(ligDict['coordList']))

                    elif ('smiles' in ligDict): # Handle simple ligands without needed specs.
                        if ligDict['smiles'] == '[H-]': # Handle hydrides
                            newdict.update({'smiles':ligDict['smiles']})
                            newdict.update({'coordList':[0]})
                            newdict.update({'ligType':'mono'})
                            newinpDict['parameters']['relax'] = False
                        elif ligDict['smiles'] == 'O':
                            newdict.update({'smiles':ligDict['smiles']})
                            newdict.update({'coordList':[0]})
                            newdict.update({'ligType':'mono'})
                        elif ligDict['smiles'] == '[OH-]':
                            newdict.update({'smiles':ligDict['smiles']})
                            newdict.update({'coordList':[0]})
                            newdict.update({'ligType':'mono'})
                        elif ligDict['smiles'] == '[O-2]':
                            newdict.update({'smiles':ligDict['smiles']})
                            newdict.update({'coordList':[0]})
                            newdict.update({'ligType':'mono'})
                            newinpDict['parameters']['relax'] = False
                        else:
                            raise ValueError('This smiles is not recognized - need coordList/ligType assignment {}!'.format(ligDict['smiles']))
                    elif ('name' in ligDict): # Read from io_ptable reference ligands!
                            if ligDict['name'].lower() in io_ptable.ligands_dict:
                                newdict = copy.deepcopy(io_ptable.ligands_dict[ligDict['name'].lower()])
                            else:
                                raise ValueError("""This ligand's {} name is not recognized. See architector.io_ptable for reference ligands.
                                 Note that this should be case-insensitive. """.format(ligDict['name']))
                    else:
                        raise ValueError('Need at least "smiles" and "coordList" for each ligand!')
                elif isinstance(ligDict,str): # Add ligand from the reference 
                    if ligDict.lower() in io_ptable.ligands_dict:
                        newdict = copy.deepcopy(io_ptable.ligands_dict[ligDict.lower()])
                    else:
                        raise ValueError("""This ligand's {} name is not recognized. See architector.io_ptable for reference ligands.
                            Note that this should be case-insensitive. """.format(ligDict))
                else:
                    raise ValueError('Unrecognized ligands definition!')
                if ('functionalizations' in ligDict):
                    tmpOBmol = io_obabel.get_obmol_smiles(newdict['smiles'],functionalizations=ligDict['functionalizations'])
                    newsmi = io_obabel.get_smiles_obmol(tmpOBmol)
                    newdict.update({'smiles':newsmi})
                newliglist.append(newdict)
        else:
            raise ValueError("inputDict['ligands'] must be a list of dictionaries currently.")

        newinpDict['ligands'] = newliglist

        #######################################################################
        ################## Finally - set parameters.  #########################
        #######################################################################

        default_parameters = {
            # Ligand to finish filling out coordination environment if underspecified.
            "fill_ligand": {"smiles":"O", "coordList":[0], "ligType":"mono"}, 
            # Secondary fill ligand will be a monodentate ligand to fill out coordination environment
            # in case the fill_ligand and specified ligands list cannot fully map.
            "secondary_fill_ligand":{"smiles":"O", "coordList":[0], "ligType":"mono"},
            # or integer index in reference to the ligand list!!
            "debug": False, # Print out addition info for debugging purposes.

            # Electronic parameters
            "metal_ox": None,
            "metal_spin": None,

            # XTB parameters.
            "xtb_solvent": 'none', # Add any named XTB solvent!
            "xtb_accuracy":1.0, # Numerical Accuracy for XTB calculations
            "xtb_electronic_temperature":300, # In K -> fermi smearing - increase for convergence on harder systems
            "xtb_max_iterations":250, # Max iterations for xtb SCF.
            "full_spin": None,
            "full_charge": None,

            # Covalent radii and vdw radii of the metal if deviations requested.
            "vdwrad_metal":vdwrad_metal,
            "covrad_metal":covrad_metal,
            "scaled_radii_factor":None, # Bookeeping if scaled vdwrad/covrad passed.
        } 

        outparams = dict()

        outparams.update(default_parameters) 

        if ('parameters' in newinpDict): # Load input parameters as changes to the defaults.
            if isinstance(newinpDict['parameters'],dict):
                outparams.update(newinpDict['parameters'])

        # Convert fill ligands information into correct graph for default ligands
        if isinstance(outparams['fill_ligand'],int):
            outparams['fill_ligand'] = newinpDict['ligands'][outparams['fill_ligand']]
        elif isinstance(outparams['fill_ligand'],str):
            if outparams['fill_ligand'] in io_ptable.ligands_dict:
                outparams['fill_ligand'] = io_ptable.ligands_dict[outparams['fill_ligand']]
            else: 
                raise ValueError('Unrecognized ligand name: {}'.format(outparams['fill_ligand']))
        
        if isinstance(outparams['secondary_fill_ligand'],str):
            if outparams['secondary_fill_ligand'] in io_ptable.ligands_dict:
                outparams['secondary_fill_ligand'] = io_ptable.ligands_dict[outparams['secondary_fill_ligand']]
            else: 
                raise ValueError('Unrecognized ligand name: {}'.format(outparams['secondary_fill_ligand']))

        if outparams['metal_ox'] is None:
            outparams['metal_ox'] = io_ptable.metal_charge_dict[metal]
        if outparams['metal_spin'] is None:
            if outparams['metal_ox'] != io_ptable.metal_charge_dict[metal]:
                # Calculate from mendeleev reference - Generally aufbau.
                outparams['metal_spin'] = mendeleev.__dict__[newinpDict['core']['metal']].ec.ionize(outparams['metal_ox']).unpaired_electrons()
            else: # Otherwise use refdict.
                outparams['metal_spin'] = io_ptable.metal_spin_dict[metal]
       
        ####### Add in missing ligands
        # Take the average of the coreTypes
        tmp_cn = int(np.round(np.mean([core_geo_class.geo_cn_dict[x] for x in coreTypes]),0))
        newLigInputDicts = newinpDict['ligands'].copy()

        nsand = np.sum([1 for x in newLigInputDicts if (x['ligType'] == 'sandwich')])
        if nsand > 0: # Currently sandwiches assigned to 3-denticity sites facial sites.
            n_fill_ligs = tmp_cn - np.sum([len(x['coordList']) for x in newLigInputDicts if x['ligType'] != 'sandwich']) - 3*nsand
        else:
            n_fill_ligs = tmp_cn - np.sum([len(x['coordList']) for x in newLigInputDicts])

        n_fill_ligs_reduced = np.floor(n_fill_ligs / len(outparams['fill_ligand']['coordList']))
        n_fill_secondary = n_fill_ligs - (n_fill_ligs_reduced)*len(outparams['fill_ligand']['coordList'])

        if n_fill_ligs < 0:
            print(n_fill_ligs, newLigInputDicts)
            raise ValueError('Error - the requested complex is over-coordinated!')

        # Populate with Fill ligand and waters.
        elif n_fill_ligs > 0:
            for _ in range(int(n_fill_ligs_reduced)):
                newLigInputDicts.append(
                    outparams['fill_ligand']
                )
            for _ in range(int(n_fill_secondary)):
                newLigInputDicts.append(
                    outparams['secondary_fill_ligand']
                )
        newinpDict['ligands'] = newLigInputDicts
        newinpDict['parameters'] = outparams
    
        return newinpDict
    else:
        raise ValueError('Input structure not recognized.')