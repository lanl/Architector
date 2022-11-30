    # def assemble_random_complex(self):
    #     """Assemble a complex with random combinations of ligand conformers."""

    #     # Variables
    #     constraintList = []

    #     mol = io_molecule.Molecule() # Initialize molecule.
    #     mol.load_ase(io_obabel.smiles2Atoms(self.coreSmiles),atom_types=[self.complexMol[0].symbol])
    #     init_sane = True
    #     # Iterate over ligands
    #     for i,ligand in enumerate(self.ligandList):
    #         if init_sane:
    #             # Find correct conformer to use
    #             conf_list = ligand.conformerList
    #             cc = np.random.choice(np.arange(len(conf_list)),1)[0]
    #             conformer = conf_list[cc]
    #             mol.append_ligand({'ase_atoms':conformer,'bo_dict':ligand.BO_dict, 
    #                                         'atom_types':ligand.atom_types})
    #             sane = mol.graph_sanity_check(params=self.parameters,assembly=True)
    #             if not sane:
    #                 init_sane = False
    #         else:
    #             if self.parameters['debug']:
    #                 print('Random conformer failed on ligand {}!'.format(self.ligandList[i-1].smiles))
    #     if self.parameters['debug']:
    #         print('Initial Sanity: ', init_sane)
    #     init_mol2str = mol.write_mol2('Charge: {} Unpaired_Electrons: {} XTB_Unpaired_Electrons: {} .mol2'.format(
    #                             int(sum(self.complexMol.get_initial_charges())),
    #                             int(io_xtb_calc.calc_suggested_spin(self.complexMol,parameters=self.parameters)),
    #                             int(sum(self.complexMol.get_initial_magnetic_moments()))), writestring=True)
    #     energy = np.inf
    #     init_energy = np.inf
    #     mol2str = None
    #     if init_sane:
    #         # Add contraints for core - Removed b/c unnecessary
    #         fixCore = ase_con.FixAtoms(indices=[0])
    #         constraintList.append(fixCore)
    #         mol.ase_atoms.set_constraint(constraintList) # Fix core metal only. 
    #         sane = io_molecule.sanity_check(mol.ase_atoms,params=self.parameters,assembly=True)
    #         if self.parameters['debug']:
    #             print('Complex sanity after adding ligands: ',sane)
    #         if sane:
    #             _ = mol.ase_atoms.copy()
    #             # Add the calculator.
    #             io_xtb_calc.set_XTB_calc(mol.ase_atoms, self.parameters, assembly=False)
    #             try:
    #                 with arch_context_manage.make_temp_directory(prefix=self.parameters['temp_prefix']) as _:
    #                     init_energy = mol.ase_atoms.get_total_energy()
    #             except Exception as e:
    #                 if self.parameters['debug']:
    #                     print(e)
    #                 sane = False
    #             if sane:
    #                 try:
    #                     with arch_context_manage.make_temp_directory(prefix=self.parameters['temp_prefix']) as _:
    #                         if self.parameters['save_trajectories']:
    #                             dyn = BFGSLineSearch(mol.ase_atoms, maxstep=3000, trajectory='temp.traj')
    #                         else:
    #                             dyn = BFGSLineSearch(mol.ase_atoms, maxstep=3000)
    #                         dyn.run(fmax=0.1)
    #                         if self.parameters['save_trajectories']:
    #                             traj = Trajectory('temp.traj')
    #                         newposits = mol.ase_atoms.get_positions() - mol.ase_atoms.get_positions()[0] # Recenter
    #                         mol.ase_atoms.set_positions(newposits)
    #                         sane1 = io_molecule.sanity_check(mol.ase_atoms, params=self.parameters) # Check close atoms or separated atoms
    #                         mol.ase_atoms.set_positions(mol.ase_atoms.get_positions()) # Set to final configuration
    #                         sane2 = mol.graph_sanity_check(params=self.parameters, assembly=False) # Check for graph-based elongated bonds
    #                         if sane1 and sane2:
    #                             sane = True
    #                         else:
    #                             sane = False
    #                         energy = mol.ase_atoms.get_total_energy()
    #                 except Exception as e: # Catch final relaxation failures.
    #                     if self.parameters['debug']:
    #                         print(e)
    #                         print('Failed final relaxation.')
    #                     sane = False
    #                     energy = 100000000
    #                 if self.parameters['save_trajectories']:
    #                     for i,ats in enumerate(traj):
    #                         self.parameters['ase_db'].write(ats,geo_step=i)
    #                 mol2str = mol.write_mol2('Charge: {} Unpaired_Electrons: {} XTB_Unpaired_Electrons: {} .mol2'.format(
    #                             int(sum(self.complexMol.get_initial_charges())),
    #                             int(io_xtb_calc.calc_suggested_spin(self.complexMol,parameters=self.parameters)),
    #                             int(sum(self.complexMol.get_initial_magnetic_moments()))), writestring=True)
    #         return mol,mol2str,energy,init_mol2str,init_energy,sane
    #     else:
    #         return mol,mol2str,energy,init_mol2str,init_energy,sane    
    
    #### Random ligand conformer generation routine. Potentially useful but not in production.
    # if inputDict['parameters']['n_lig_combos']>1:
    #     mol2strs = []
    #     mols = []
    #     xtb_energies1 = []
    #     init_mol2strs = []
    #     init_xtb_energies = []
    #     for k in range(2*inputDict['parameters']['n_lig_combos']):
    #         mol,mol2str,energy,init_mol2str,init_energy,sane = structs[i].assemble_random_complex()
    #         if sane:
    #             mols.append(mol)
    #             mol2strs.append(mol2str)
    #             xtb_energies1.append(energy)
    #             init_mol2strs.append(init_mol2str)
    #             init_xtb_energies.append(init_energy)
    #         if len(mol2strs) == inputDict['parameters']['n_lig_combos']:
    #             break
    #     for k in range(len(mol2strs)):
    #         outkey = keys[i] + '_random_complex_' + str(k)
    #         ordered_conf_dict[outkey] = {'ase_atoms':mol.ase_atoms, 
    #             # 'complex':structs[i], # Debug
    #             'total_charge':np.sum(mols[k].ase_atoms.get_initial_charges()),
    #             'n_unpaired_electrons': np.sum(mols[k].ase_atoms.get_initial_magnetic_moments()),
    #             'calc_n_unpaired_electrons': io_xtb_calc.calc_suggested_spin(structs[i].complexMol,
    #                                                                         parameters=inputDict['parameters']),
    #             'metal_ox':inputDict['parameters']['metal_ox'],
    #             'init_energy':init_xtb_energies[k],
    #             'energy':xtb_energies1[k],
    #             'mol2string':mol2strs[k], 'init_mol2string':init_mol2strs[k],
    #             'energy_sorted_index': energy_sorted_inds[i],
    #             'inputDict':inputDict
    #             }
    #         if inputDict['parameters']['return_full_complex_class']: # Return whole complex class (all ligand geometries!)
    #             ordered_conf_dict[outkey].update({'full_complex_class':structs[i]
    #                 })
    #         if inputDict['parameters']['return_timings']:
    #             tdict.update({'core_preprocess_time':core_preprocess_time,
    #                         'symmetry_preprocess_time':symmetry_preprocess_time,
    #                         'total_liggen_time':np.sum([x.total_liggen_time for x in structs[i].ligandList]),
    #                         'total_complex_assembly_time':structs[i].assemble_total_time,
    #                         'final_relaxation_time':structs[i].final_eval_total_time,
    #                         'sum_total_conformer_time_spent':fin_time2 - int_time1
    #                     })