"""
Build complexes based on user input.

Developed by Dan Burrill and Michael Taylor
"""

# Imports
import copy
import time
import numpy as np
import shutil
from collections import OrderedDict

import architector.io_process_input as io_process_input
import architector.io_obabel as io_obabel
import architector.io_lig as io_lig
import architector.io_ptable as io_ptable
import architector.io_molecule as io_molecule
import architector.io_align_mol as io_align_mol
import architector.io_crest as io_crest
import architector.io_symmetry as io_symmetry

from architector.io_calc import CalcExecutor

class Ligand:
    """Class to contain all information about a ligand including conformers."""

    def __init__(self, smiles, ligcoordList, corecoordList, core, ligGeo, ligcharge,
                covrad_metal=None, vdwrad_metal=None, debug=False):
        """Set up initial variables for ligand and run conformer generation routines.

        Parameters
        -----------
        smiles : str
            Ligand Smiles
        ligcoordList : list (int)
            List of metal-coordinating indices for the ligand tied to indices of core vectors.
        corecoordList : list (float)
            Core vectors definitions (2 D array with denticityX3 shape)
        core : str
            Metal identity
        ligGeo : str
            The binned ligand geometry
        ligcharge : float
            Ligand charge if determined
        covrad_metal : float, optional
            Covalent radii of the metal, default values from io_ptable.rcov1
        vdwrad_metal : float, optional
            VDW radii of the metal, default values from io_ptable.rvdw
        debug: bool, optional
            debug turns on/off output text/comments.
        """
        # Set variables
        self.smiles = smiles
        self.ligcoordList = ligcoordList
        self.corecoordList = corecoordList
        self.metal = core
        self.geo = ligGeo 
        self.BO_dict = None
        self.atom_types = None
        self.out_ligcoordLists = None
        self.charge = ligcharge
        self.liggen_start_time = time.time()
        # Generate conformations
        if debug:
            print("GENERATING CONFORMATIONS for {}".format(smiles))
        conformers, rotscores, tligcoordList, relax, bo_dict, atypes, rotlist = io_lig.find_conformers(self.smiles, 
                                                        self.ligcoordList, 
                                                        self.corecoordList, 
                                                        metal=self.metal,
                                                        ligtype=self.geo,
                                                        covrad_metal=covrad_metal,
                                                        vdwrad_metal=vdwrad_metal,
                                                        debug=debug
                                                        )
        if len(conformers) > 0:
            self.conformerList = conformers
            self.conformerRotScore = rotscores
            self.rotList = rotlist
            self.out_ligcoordLists = tligcoordList
            self.selectedConformer = self.conformerList[0]
            self.exists = True 
            self.relax = relax
            self.BO_dict = bo_dict
            self.atom_types = atypes
            self.liggen_end_time = time.time()
            if debug:
                print("CONFORMERS GENERATED for {}".format(smiles))
        else:
            if debug:
                print('No Valid Conformers Generated for {}!'.format(smiles))
            self.exists = False
            self.relax = False
            self.conformerList = []
            self.out_ligcoordLists = []
            self.rotLists = []
            self.selectedConformer = None
            self.liggen_end_time = time.time()
            self.BO_dict = dict()
            self.atom_types = []
        self.total_liggen_time = self.liggen_end_time - self.liggen_start_time

class Complex:
    """Class that contains all information and functions regarding a complex."""

    def __init__(self, coreSmiles, coordList, ligandList, parameters):
        """Initialize complex and call assemble functions.

        Parameters
        -----------
        coreSmiles : str
            Core Smiles
        coordList : list (int)
            Core vectors definitions (2 D array with denticityX3 shape)
        ligandList : list (Ligand)
            List of generated Ligand classes
        parameters : dict 
            All input parameters dictionary
        """
        # Set variables
        self.coreSmiles = coreSmiles
        self.coordList = coordList
        self.ligandList = ligandList
        self.parameters = parameters
        self.calculator = None
        self.save = True # Save corresponds to whether calculations completed or failed.
        self.index = 0 # Index keeps track of which index conformer this is.
        self.initMol = None
        self.initEnergy = None
        self.finalEvalTotalTime = None
        self.allLigandsGood = True
        self.assembled = False

        # Initialize Atoms object for complex
        init_atoms = io_obabel.smiles2Atoms(self.coreSmiles, addHydrogens=False)
        init_atoms.set_initial_charges([parameters['metal_ox']])
        init_atoms.set_initial_magnetic_moments([parameters['metal_spin']])

        self.complexMol = io_molecule.convert_io_molecule(init_atoms)
        self.assembleMol = None

        # Assemble complex
        if self.parameters['debug']:
            print("ASSEMBLING COMPLEX")
        self.assemble_start_time = time.time()
        self.assemble_complex()
        self.assemble_end_time = time.time()
        self.assemble_total_time = self.assemble_end_time - self.assemble_start_time

    def assemble_complex(self):
        """Assemble the complex one ligand at a time.

        Will currently add each ligand in order and evaluate with XTB to find the "best"
        Ligand for each coordination site.
        """
        # Variables

        self.initMol = io_molecule.convert_io_molecule(self.complexMol)

        for i,ligand in enumerate(self.ligandList):
            if self.parameters['debug']:
                print("LIGAND: {}".format(ligand.smiles))
                # Find correct conformer to use
                print("FINDING CORRECT CONFORMER")
            conformerList = ligand.conformerList
            rot_vals = ligand.conformerRotScore
            # Compute conformer efficacy by XTB-energy and fit to binding sites.
            bestConformer, assembled = self.compute_conformer_efficacy(conformerList,
                                                                rot_vals,
                                                                ligand)
            if assembled:
                self.complexMol.append_ligand({'ase_atoms':bestConformer,'bo_dict':ligand.BO_dict, 
                                            'atom_types':ligand.atom_types})
                self.initMol.append_ligand({'ase_atoms':bestConformer,'bo_dict':ligand.BO_dict, 
                                            'atom_types':ligand.atom_types})
            else: # Check for failures - do not evaluate.
                self.allLigandsGood = False
                if self.parameters['debug']:
                    print('Ligand {} was not able to be generated!'.format(self.ligandList[i-1].smiles))
                break                
        if self.parameters['debug']:
            print('Initial ligand geometry sanity: ', self.allLigandsGood)
        if self.allLigandsGood: # Only perform total energy if all ligands sane and no overlaps
            self.assembled = True

    def compute_conformer_efficacy(self,conformerList,rot_vals,ligand):
        """Conformer efficiency currently calculated by GFN-FF in XTB for acceleration
        
        Now possible to specify method for comparing ligands.

        Parameters
        ---------
        conformerList : list (ase.atoms.Atoms)
            Ligand conformers
        rot_vals : list (float)
            rotational loss values for ligands to assigned binding sites
        ligand : Ligand class
            ligand information including atom types and bond order (BO) dict.

        Returns
        --------
        bestConformer : ase.atoms.Atoms
            best ligand structure
        assembled : bool
            Whether the addition of any of the ligand conformers was successful
        """
        best_val = np.inf
        bestConformer = conformerList[0]
        assembled = False
        for i,conformer in enumerate(conformerList):# Try and use XTB
            tmp_molecule = io_molecule.convert_io_molecule(self.initMol)
            tmp_molecule.append_ligand({'ase_atoms':conformer,'bo_dict':ligand.BO_dict, 
                            'atom_types':ligand.atom_types})
            if self.parameters['debug']:
                print(tmp_molecule.write_mol2('cool{}.mol2'.format(i),writestring=True))
            out_eval = CalcExecutor(tmp_molecule,assembly=True, 
                                    parameters=self.parameters,
                                    init_sanity_check=True)
            if out_eval.successful:
                Eval = out_eval.energy*(1/rot_vals[i]) # Bias to lower rotational loss values
                if Eval < best_val and out_eval.successful: 
                    assembled = True
                    bestConformer = conformer
                    best_val = Eval
            elif (not out_eval.successful) and (self.parameters['debug']):
                print('Ligand {} failed xtb/uff or overlapped.'.format(i))
        return bestConformer, assembled
            
    def final_eval(self,single_point=False):
        """final_eval perform final evaulation of full complex conformer with XTB.
        
        Involves either a relaxation or not for each "sane" conformer.

        Parameters
        ----------
        single_point : bool, optional
            Perform only a singlepoint calculation?, by default False
        """
        self.final_start_time = time.time()
        self.initMol.dist_sanity_checks(params=self.parameters,assembly=single_point)
        self.initMol.graph_sanity_checks(params=self.parameters,assembly=single_point)
        if self.assembled:
            if self.parameters['debug']:
                print("Final Evaluation - Opt Molecule/Single point")
            self.calculator = CalcExecutor(self.complexMol,parameters=self.parameters,
                                            final_sanity_check=self.parameters['full_sanity_checks'],
                                            relax=single_point,assembly=single_point)
            if self.parameters['debug'] and (not self.calculator.successful):
                print('Failed final relaxation. - Retrying with UFF/XTB')
                print(self.initMol.write_mol2('cool.mol2', writestring=True))
            # Retry with 2 step optimization -> first do UFF -> then do the requested method.
            if (not self.calculator.successful):
                tmp_relax = CalcExecutor(self.complexMol,method='UFF',fix_m_neighbors=False,relax=single_point)
                self.calculator = CalcExecutor(tmp_relax.mol,parameters=self.parameters,
                                                final_sanity_check=self.parameters['full_sanity_checks'],
                                                relax=single_point)
        else: # Ensure calculation object at least exists
            self.calculator = CalcExecutor(self.complexMol,method='UFF',fix_m_neighbors=False,relax=False)
            self.calculator.successful = False
        self.final_end_time = time.time()
        self.final_eval_total_time = self.final_end_time - self.final_start_time
        if self.calculator:
            self.complexMol = self.calculator.mol
    
    def swap_metals_back(self,in_metal=None):
        """swap_metals_back function to swap the metals back to original states.

        Parameters
        ----------
        in_metal : str, optional
            if an metal was passed in, by default None
        """
        self.initMol.ase_atoms[0].symbol = copy.deepcopy(self.complexMol.ase_atoms.symbols[0])
        self.initMol.atom_types[0] = copy.deepcopy(self.complexMol.ase_atoms.symbols[0])
        string = self.initMol.write_mol2('init_geo', writestring=True)
        self.init_geo_swapped_metal = copy.deepcopy(string)
        if in_metal: # Switch to the original metal.
            self.initMol.ase_atoms[0].symbol = in_metal
            self.initMol.atom_types[0] = in_metal
            self.complexMol.ase_atoms[0].symbol = in_metal
            self.complexMol.atom_types[0] = in_metal
        elif self.parameters['in_metal']: # Switch to the original metal.
            self.initMol.ase_atoms[0].symbol = self.parameters['in_metal']
            self.initMol.atom_types[0] = self.parameters['in_metal']
            self.complexMol.ase_atoms[0].symbol = self.parameters['in_metal']
            self.complexMol.atom_types[0] = self.parameters['in_metal']
        elif self.parameters['is_actinide']: # Convert back to actinide if needed
            self.initMol.ase_atoms[0].symbol = self.parameters['original_metal']
            self.initMol.atom_types[0] = self.parameters['original_metal']
            self.complexMol.ase_atoms[0].symbol = self.parameters['original_metal']
            self.complexMol.atom_types[0] = self.parameters['original_metal']


def gen_aligned_complex(newLigInputDicts, 
                        ligandDict,
                        inputDict, 
                        ligLists, 
                        coreCoordList,
                        coreType):
    """gen_aligned_complex 

    Parameters
    ----------
    newLigInputDicts : dict
        Dictionary of ligand/coordination site assignments from io_symmetry
    ligandDict : dict
        ligand dictionary from previous generations -> used to skip multiple runnings of ligand conformer generations.
    inputDict : dict
        total input dictionary
    ligLists : list (list)
       Coordination assignments for the ligand
    coreCoordList : list (list)
        Core coordination vectors
    coreType : str
        what type of core is this -> used mostly for debugging

    Returns
    -------
    complexClass : Complex
        complex class containing all information and 3D structure!
    ligandDict : dict
        output ligand geometries and conformers for this structure for use in future structures generated.
    """
    ligandList = []
    for i,ligand in enumerate(newLigInputDicts):
        # Get ligand smiles
        ligandSmiles = ligand["smiles"]
        ligGeo = ligand['ligType']
        ligCharge = ligand['ligCharge']
        ligcons = '_'.join(sorted([str(x[0]) for x in ligLists[i]]))
        ligid = ligandSmiles + ligcons

        # Generate conformations if not already done
        if (ligid not in ligandDict):
            # Generate Ligand class
            ligandClass = Ligand(ligandSmiles,
                                ligLists[i],
                                coreCoordList,
                                inputDict["core"]["smiles"].strip('[').strip(']'),
                                ligGeo,
                                ligCharge,
                                covrad_metal = inputDict['parameters']['covrad_metal'],
                                vdwrad_metal = inputDict['parameters']['vdwrad_metal'],
                                debug = inputDict['parameters']['debug']
                                )
            # Store results
            ligandDict[ligid] = ligandClass
            ligandList.append(ligandClass)
        else:
            # Use stored result - > re-apply rotations to match desired sites.
            ligandCopy = copy.deepcopy(ligandDict[ligid])
            ligandCopy.ligcoordList = ligLists[i]
            ligandCopy.corecoordList = coreCoordList # DO NOT DELETE THIS LINE
            newligconfList = []
            ligconfVals = []
            for j,lig in enumerate(ligandCopy.conformerList):
                new_ligcoordList = [[val[0],ligandCopy.ligcoordList[k][1]] for k,val in enumerate(ligandCopy.out_ligcoordLists[j])]
                rot_angle = ligandCopy.rotList[j]
                if rot_angle != 0: # Apply same rotations.
                    newconf, rotscore, sane = io_lig.set_position_align(lig, 
                                                    new_ligcoordList, 
                                                    ligandCopy.corecoordList,
                                                    debug=inputDict['parameters']['debug'],
                                                    rot_coord_vect=True,
                                                    rot_angle=rot_angle)
                else:
                    newconf, rotscore, sane = io_lig.set_position_align(lig, 
                                                                        new_ligcoordList, 
                                                                        ligandCopy.corecoordList,
                                                                        debug=inputDict['parameters']['debug'])
                if sane:
                    ligconfVals.append(rotscore)
                    newligconfList.append(newconf)
                else:
                    if inputDict['parameters']['debug']:
                        print('Conformer sucks!')
            ligandCopy.conformerList = newligconfList
            ligandCopy.conformerRotScore = ligconfVals
            ligandList.append(copy.deepcopy(ligandCopy))
    if all([x.exists for x in ligandList]): # Check that all ligands were able to generate at least one conformer
        coreSmiles = inputDict["core"]["smiles"]
        coreCoordList = coreCoordList
        complexClass = Complex(coreSmiles, coreCoordList, ligandList, inputDict['parameters'])
        complexClass.final_eval(single_point=True) # Key here - have singlepoints during assembly be defined by assembly params.
        if inputDict['parameters']['debug']:
            print('Complex class generated: ', complexClass.calculator.successful)
        if (not complexClass.assembled) or (not complexClass.calculator.successful) or (not complexClass.calculator.mol.dists_sane):
            if inputDict['parameters']['debug']:
                print('Generated geometry is not sane or XTB failed for coreType: ', coreType)
            complexClass = None
    else:
        complexClass = None
        if inputDict['parameters']['debug']:
            print('At least one ligand was not able to be generated when mapped to this core.')

    return complexClass, ligandDict

# Functions
def complex_driver(inputDict1,in_metal=False):
    """complex_driver Driver function for complex construction.

    Parameters
    ----------
    inputDict1 : dict
        inputDictionary directly from the user.
    in_metal : bool, optional
        if a metal was swapped, by default False

    Returns
    -------
    conf_dict : dict 
        conformers dictionary of a given complex.
    inputDict : dict
        inparse-processed input dictionary for the complex
    core_preprocess_time : float
        time spent generating the core dictionary and binding site assignments. (seconds)
    symmetry_preprocess_time : float
        time spent processing the symmetries of the complex (seconds)
    int_time1 : float
        starting time of the full complex generation (seconds)
    """
    # Process input Dictionary
    start_time0 = time.time()
    inputDict = io_process_input.inparse(inputDict1)
    inputDict['parameters']['in_metal'] = in_metal
    fin_time0 = time.time()
    core_preprocess_time = fin_time0 - start_time0
    
    coreTypes = inputDict['coreTypes']
    core_geo_class = inputDict['core_geo_class']

    ligandDict = {}
    conf_dict = {}

    if len(coreTypes) > 0: # Catch where no coretypes assigned
        for coreType in coreTypes:
            int_time1 = time.time()
            # Re-initialize ligands dict to match new core geometry -> save monodentates!!!!!
            if len(ligandDict) > 0:
                newligDict = dict()
                for key,oldligclass in ligandDict.items():
                    if oldligclass.geo == 'mono':
                        newligDict[key] = oldligclass
                ligandDict = newligDict
            else: # Generate from scratch
                ligandDict = {} # 
            
            coreCoordList = core_geo_class.geometry_dict[coreType]

            # Assign con atoms based on all ligands
            newLigInputDicts, all_liglists, good = io_symmetry.select_cons(inputDict["ligands"],
                                                            coreType, core_geo_class, inputDict['parameters']
                                                            )
            if inputDict['parameters']['debug']:
                print('Assigned LigCons ->')
                print('LigLists:', all_liglists) 
                print('coreCoordList:', coreCoordList)

            fin_time1 = time.time()
            symmetry_preprocess_time = fin_time1 - int_time1
            if good: # Try first one
                # Test selected ligand symmetries
                # -> don't generate new conformers/shift ligandDict for different symmetries
                out_complexlist = []
                out_energies = []
                for i,ligLists in enumerate(all_liglists): 
                    if i == 0: # Generate and save ligandDict from first conformer
                        complexClass, ligandDict = gen_aligned_complex(
                            newLigInputDicts, 
                            ligandDict,
                            inputDict, 
                            ligLists, 
                            coreCoordList,
                            coreType)
                        if (complexClass is None): # Catch when generated conformer is invalid.
                            complexClass = False
                        else:
                            out_complexlist.append(complexClass)
                            out_complexlist[-1].index = i
                            out_energies.append(complexClass.calculator.energy)
                            if inputDict['parameters']['return_only_1']:
                                break
                    else: # Use the first conformer ligand structures to map to other symmetries
                        tcomplexClass, _ = gen_aligned_complex(
                                newLigInputDicts,  
                                ligandDict,
                                inputDict,
                                ligLists, 
                                coreCoordList,
                                coreType)
                        if (tcomplexClass is not None):
                            complexClass = tcomplexClass
                            out_complexlist.append(complexClass)
                            out_complexlist[-1].index = i
                            out_energies.append(complexClass.calculator.energy)
                            if inputDict['parameters']['return_only_1']:
                                break
                if not isinstance(complexClass, bool): # Catch cases where no conformation generated.
                    order = np.argsort(out_energies)
                    for ind,j in enumerate(order[0:inputDict['parameters']['n_conformers']]):
                        tmp_conformer = out_complexlist[j]
                        if any([(not x.relax) for x in tmp_conformer.ligandList]):
                            if inputDict['parameters']['debug']:
                                print('Warning This complex is likely strange because of failures of MMFF94 or distance geometry!')
                                print('Defaulting to single point evaluation.')
                        else: # Do the final relaxation (if good) on each conformer to save!
                            tmp_conformer.final_eval()
                        if inputDict['parameters']['debug']:
                            print('Complex Distances Sane: ', tmp_conformer.complexMol.dists_sane)
                        spin_n_unpaired = np.sum(tmp_conformer.complexMol.xtb_uhf)
                        tot_charge = np.sum(tmp_conformer.complexMol.xtb_charge)
                        if tmp_conformer.calculator is not None:
                            if tmp_conformer.complexMol.dists_sane and tmp_conformer.calculator.successful: # Check sanity after
                                conf_dict.update({coreType + '_' + str(ind) + '_nunpairedes_' + \
                                    str(int(spin_n_unpaired))+'_charge_'+str(int(tot_charge)):tmp_conformer})
                                if inputDict['parameters']['return_only_1']:
                                        return conf_dict,inputDict,core_preprocess_time,symmetry_preprocess_time,int_time1
                            elif tmp_conformer.initMol.dists_sane and inputDict['parameters']['save_init_geos']:
                                tmp_conformer.calculator.energy = 10000 # Set to high energy.
                                conf_dict.update({coreType + '_' + str(ind) + '_nunpairedes_' + \
                                    str(int(spin_n_unpaired))+'_charge_'+str(int(tot_charge))+'_init_only':tmp_conformer})
                                if inputDict['parameters']['return_only_1']:
                                        return conf_dict,inputDict,core_preprocess_time,symmetry_preprocess_time,int_time1
                        else:
                            if inputDict['parameters']['debug']:
                                print('Skipping complex due to no calculator assignment -> not assembled .')
                else:
                    if inputDict['parameters']['debug']:
                        print('Complex not generated due to lack of ability for ligand to map to core.')
            else:
                if inputDict['parameters']['debug']:
                    print('No coordination environment avaiable for this ligand combination')
        return conf_dict,inputDict,core_preprocess_time,symmetry_preprocess_time,int_time1
    else:
        return {},inputDict,0,0,0
    
def build_complex_driver(inputDict1,in_metal=False):
    """build_complex_driver overall driver building of the complex

    Parameters
    ----------
    inputDict1 : dict
        inputDictionary directly from the user.
    in_metal : bool, optional
        if a metal was swapped, by default False

    Returns
    -------
    ordered_conf_dict : dict
        Conformer dictionary with stored values if generation successful.
    """
    conf_dict,inputDict,core_preprocess_time,symmetry_preprocess_time,int_time1 = complex_driver(inputDict1=inputDict1,in_metal=in_metal)
    if len(conf_dict) == 0:
        if inputDict['parameters']['debug']:
            print('No possible geometries for the input ligand/coreType(s) combination.')
        ordered_conf_dict = conf_dict
    else:
        ordered_conf_dict = OrderedDict()
        xtb_energies = []
        energy_sorted_inds = []
        keys = []
        structs = []
        mol2strings = []
        init_mol2strings = []
        for key,val in conf_dict.items():
            xtb_energies.append(val.calculator.energy)
            keys.append(key)
            val.swap_metals_back(in_metal=in_metal)
            structs.append(val)
            if inputDict['parameters']['save_init_geos']:
                init_mol2strings.append(val.initMol.write_mol2('{}'.format(key), writestring=True))
            else:
                init_mol2strings.append(None)
            energy_sorted_inds.append(val.index) # Save energy sorted index for reference.
            mol2strings.append(val.complexMol.write_mol2('{}'.format(key), writestring=True))
        order = np.argsort(xtb_energies)
        # Iterate through all structures and check/remove duplicate structures.
        # Remove extra classes that we don't need to persist
        del inputDict['core_geo_class']
        del inputDict['parameters']['ase_db'] 
        for ind,i in enumerate(order):
            iscopy = False
            if (ind > 0) and (not inputDict['parameters']['skip_duplicate_tests']): # Check for copies
                for key,val in ordered_conf_dict.items():
                    # if ('_init_only' in key) or ('_init_only' in keys[i]): # Do not do duplicate test on init_only structures.
                    #     continue
                    # else:
                    _, rmsd_full, _ = io_align_mol.calc_rmsd(mol2strings[i],val['mol2string'],coresize=10)
                    if (rmsd_full < 0.5):
                        iscopy = True
                        break
                    rmsd_core, _, _ = io_align_mol.calc_rmsd(mol2strings[i],val['mol2string'])
                    if (rmsd_core < 0.7) and np.isclose(val['energy'],xtb_energies[i],atol=0.1):
                        iscopy = True
                        break
                if (not iscopy):
                    ordered_conf_dict[keys[i]] = {'ase_atoms':structs[i].complexMol.ase_atoms,
                            'total_charge':int(structs[i].complexMol.charge),
                            'xtb_n_unpaired_electrons': structs[i].complexMol.xtb_uhf,
                            'xtb_total_charge':int(structs[i].complexMol.xtb_charge),
                            'calc_n_unpaired_electrons': structs[i].complexMol.uhf,
                            'metal_ox':inputDict['parameters']['metal_ox'],
                            'init_energy':structs[i].calculator.init_energy,
                            'energy':xtb_energies[i],
                            'mol2string':mol2strings[i], 'init_mol2string':init_mol2strings[i],
                            'energy_sorted_index': energy_sorted_inds[i],
                            'inputDict':inputDict
                            }
            else:
                ordered_conf_dict[keys[i]] = {'ase_atoms':structs[i].complexMol.ase_atoms, 
                        'total_charge':int(structs[i].complexMol.charge),
                        'xtb_n_unpaired_electrons': structs[i].complexMol.xtb_uhf,
                        'calc_n_unpaired_electrons': structs[i].complexMol.uhf,
                        'xtb_total_charge':int(structs[i].complexMol.xtb_charge),
                        'metal_ox':inputDict['parameters']['metal_ox'],
                        'init_energy':structs[i].calculator.init_energy,
                        'energy':xtb_energies[i],
                        'mol2string':mol2strings[i], 'init_mol2string':init_mol2strings[i],
                        'energy_sorted_index': energy_sorted_inds[i],
                        'inputDict':inputDict
                        }
            if (not iscopy) and inputDict['parameters']['return_timings']:
                tdict = ordered_conf_dict[keys[i]]
                fin_time2 = time.time()
                tdict.update({'core_preprocess_time':core_preprocess_time,
                    'symmetry_preprocess_time':symmetry_preprocess_time,
                    'total_liggen_time':np.sum([x.total_liggen_time for x in structs[i].ligandList]),
                    'total_complex_assembly_time':structs[i].assemble_total_time,
                    'final_relaxation_time':structs[i].final_eval_total_time,
                    'sum_total_conformer_time_spent':fin_time2 - int_time1
                })
                ordered_conf_dict[keys[i]] = tdict
            if (not iscopy) and inputDict['parameters']['return_full_complex_class']: # Return whole complex class (all ligand geometries!)
                tdict = ordered_conf_dict[keys[i]]
                tdict.update({'full_complex_class':structs[i]
                })
                ordered_conf_dict[keys[i]] = tdict
    return ordered_conf_dict

def build_complex(inputDict):
    """build_complex build a complex!
    main function for architector!!!!
    Attempts a default radii, larger radii, and smaller radii build_complex_driver
    run for each pass in attempt to converge to assigned spin/charge states.


    Parameters
    ----------
    inputDict : dict
        input dictionary

    Returns
    -------
    ordered_conf_dict : dict
        ordered structures and outputs as valid
    """
    ordered_conf_dict = build_complex_driver(inputDict)
    # Try larger radii generation for multidentate complexes if no complexes generated in an attempt to get at high-spin
    # > Covalent radii typically understimated for higher spin conformations
    in_metal = inputDict['parameters']['original_metal']
    if 'mol2string' in inputDict:
        tmp_inputDict = io_process_input.inparse(inputDict)
    else:
        tmp_inputDict = io_process_input.inparse(inputDict)
    if (len([x for x in ordered_conf_dict.keys() if ('_init_only' not in x)]) == 0) and \
       (max([len(x['coordList']) for x in tmp_inputDict['ligands']]) > 2):
        newinpdict = io_ptable.map_metal_radii(tmp_inputDict,larger=True) # Run with larger radii
        if tmp_inputDict['parameters']['debug']:
            print('Trying with larger scaled metal radii.')
        temp_ordered_conf_dict = build_complex_driver(newinpdict,in_metal=in_metal)
        newdict_append = dict()
        for key,val in temp_ordered_conf_dict.items():
            newdict_append[key+'_larger_scaled'] = val
        ordered_conf_dict.update(newdict_append)
        if (len([x for x in ordered_conf_dict.keys() if ('_init_only' not in x)]) > 0):
            try_smaller = False
            if tmp_inputDict['parameters']['debug']:
                print('Succeeded with larger scaled metal radii!')
        else:
            try_smaller=True
            if tmp_inputDict['parameters']['debug']:
                print('No possible structures for this structure even with larger radii structure.')
        if try_smaller: # Run with smaller radii
            if tmp_inputDict['parameters']['debug']:
                print('Trying with smaller scaled metal radii.')
            newinpdict = io_ptable.map_metal_radii(tmp_inputDict,larger=False)
            temp_ordered_conf_dict = build_complex_driver(newinpdict,in_metal=in_metal)
            newdict_append = dict()
            for key,val in temp_ordered_conf_dict.items():
                newdict_append[key+'_smaller_scaled'] = val
            ordered_conf_dict.update(newdict_append)
            if (len([x for x in ordered_conf_dict.keys() if ('_init_only' not in x)]) > 0):
                if tmp_inputDict['parameters']['debug']:
                    print('Succeeded with smaller scaled metal radii!')
            else:
                if tmp_inputDict['parameters']['debug']:
                    print('No possible structures for this structure even with smaller radii structure.')
    # Final Reorder
    if len(ordered_conf_dict) > 0:
        out_ordered_conf_dict = OrderedDict()
        xtb_energies = []
        keys = []
        vals = []
        for key,val in ordered_conf_dict.items():
            xtb_energies.append(val['energy'])
            keys.append(key)
            vals.append(val)
        order = np.argsort(xtb_energies)
        for j,i in enumerate(order):
            if tmp_inputDict['parameters']['crest_sampling'] and (j == 0): # Run crest sampling on lowest energy isomer!
                samples,energies = io_crest.crest_conformers(vals[i]['mol2string'],solvent=tmp_inputDict['parameters']['xtb_solvent'])
                vals[i].update({'crest_conformers':samples,'crest_energies':energies})
                vals[i].update({'energy':min(energies)})
                tmpmol = io_molecule.convert_io_molecule(vals[i]['mol2string'])
                posits = io_molecule.convert_io_molecule(samples[0]).ase_atoms.get_positions()
                tmpmol.ase_atoms.set_positions(posits)
                vals[i].update({'mol2string':tmpmol.write_mol2('Crest_Min_Energy', writestring=True)})
            out_ordered_conf_dict[keys[i]] = vals[i]
    else:
        out_ordered_conf_dict = dict()
    # At the end move the .json files generated to the cwd -> done to save time!
    if tmp_inputDict['parameters']['save_trajectories'] or tmp_inputDict['parameters']['dump_ase_atoms']:
        shutil.copy(tmp_inputDict['parameters']['ase_db_tmp_name'],
                tmp_inputDict['parameters']['ase_atoms_db_name'])
    return out_ordered_conf_dict

def build_complex_2D(inputDict):
    """build_complex_2D
    Rather than do full 3D construction - do 2D building of the complex (orders of magnitude faster!)

    Parameters
    ----------
    inputDict : dict
        input dictionary in the same architector style
    """
    inputDict = io_process_input.inparse_2D(inputDict)

        # Initialize Atoms object for complex
    complexMol = io_obabel.smiles2Atoms(inputDict['core']['smiles'], addHydrogens=False)
    charge = inputDict['parameters']['metal_ox']

    mol = io_molecule.Molecule() # Initialize molecule.
    mol.load_ase(complexMol.copy(),atom_types=[complexMol[0].symbol])

    # Assemble complex
    for i,ligand in enumerate(inputDict['ligands']):
        obmollig = io_obabel.get_obmol_smiles(ligand['smiles'],addHydrogens=True,neutralize=False,build=False)
        ligcharge = obmollig.GetTotalCharge()
        charge = charge + ligcharge
        bestConformer = io_obabel.convert_obmol_ase(obmollig,posits=None,set_zero=True)
        io_obabel.add_dummy_metal(obmollig,ligand['coordList'])
        bo_dict, atypes = io_obabel.get_OBMol_bo_dict_atom_types(obmollig)

        mol.append_ligand({'ase_atoms':bestConformer,'bo_dict':bo_dict, 
                                    'atom_types':atypes})
    
    # Charge -> charges already assigned to components during assembly
    if (inputDict['parameters']['full_charge'] is not None):
        charge = inputDict['parameters']['full_charge']

    mol_charge = charge

    symbols = mol.ase_atoms.get_chemical_symbols()
    metals = [x for x in symbols if x in io_ptable.all_metals]

    f_in_core = False
    
    if len(metals) == 1:
        if metals[0] in io_ptable.heavy_metals:
            f_in_core = True
    else:
        if inputDict['parameters']['debug']:
            print('No metals - continuing anyway!')
    
    
    # Handle spin / magnetism
    even_odd_electrons = (np.sum([atom.number for atom in mol.ase_atoms])-mol_charge) % 2
    if (inputDict['parameters']['full_spin'] is not None):
        uhf = inputDict['parameters']['full_spin']
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
    else:
        uhf = inputDict['parameters']['metal_spin'] # Metal spin set by io_process_input to defaults.
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
    xtb_unpaired_electrons = copy.copy(uhf)
    xtb_charge = copy.copy(mol_charge)

    if f_in_core: # F in core assumes for a 3+ lanthanide with 11 valence electrons for XTB
        xtb_charge = charge + (3 - inputDict['parameters']['metal_ox'])
        even_odd_electrons = (np.sum([atom.number for atom in mol.ase_atoms]))
        even_odd_electrons = even_odd_electrons - io_ptable.elements.index(metals[0]) + 11 - xtb_charge
        even_odd_electrons = even_odd_electrons % 2
        if (even_odd_electrons == 0):
            xtb_unpaired_electrons = 0
        else:
            xtb_unpaired_electrons = 1

    mol.xtb_uhf = xtb_unpaired_electrons
    mol.xtb_charge = xtb_unpaired_electrons
    mol.uhf = uhf
    mol.charge = mol_charge

    return {'mol2string':mol.write_mol2('2D_Mol:', writestring=True),'input_dict':inputDict}

# Main
if (__name__ == '__main__'):
    # Variables
    inputDict = {
                    "core": {"smiles":"[Fe]",
                             "coreType":'octahedral',
                            }, 
                    "ligands":[
                        {"smiles":"n1ccccc1-c2ccccn2",
                                   "coordList":[0, 1],
                                   'ligType':'bi_cis'},
                                   ], # If core CN specified and ligand locations remaining -> populate with water
                    "parameters":{}
                }

    # Build complex
    complexMol_dict = build_complex(inputDict)
