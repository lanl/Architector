"""
Py3Dmol install: (works in both Python2/3 conda environments)
conda install -c conda-forge py3dmol 
Some Documentation: https://pypi.org/project/py3Dmol/
3DMol.js backend: http://3dmol.csb.pitt.edu/index.html

Visualization routine for handling jupyter notebook use of architector.

Normal modes code adapted from: https://github.com/duerrsimon/normal-mode-jupyter

Developed by Michael Taylor
"""
import math as m
import numpy as np
import ase

import py3Dmol

import architector
from architector.io_molecule import convert_io_molecule
import architector.io_ptable as io_ptable

def type_convert(structures):
    """Handle multiple types of structures passed. List of xyz, mol2 files,
    or list of xyz, mol2strings.

    Parameters
    ----------
    structures : list
        Structures you want visualized: can either be a list or individual:
        mol2 strings, mol2 files, xyz strings, xyz files, or mol3D objects
    """
    outlist = []
    if isinstance(structures,str):
        out = convert_io_molecule(structures)
        if isinstance(out,(np.ndarray,list)): # Read in traj.
            structures=out
        else:
            structures = [out]
    elif isinstance(structures,(ase.atoms.Atoms,architector.io_molecule.Molecule)):
        structures = [convert_io_molecule(structures)]
    elif isinstance(structures,dict):
        try:
            structures = [val['mol2string'] for key,val in structures.items()]
        except:
            raise ValueError('Not recognized type for this dictionary to visualize!')
    else: # Convert other array-like arguments to a list.
        structures = list(structures)
    if isinstance(structures,list):
        for i,x in enumerate(structures):
            try: 
                mol = convert_io_molecule(x)
                outlist.append(mol)
            except:
                raise ValueError('Not Recognized Structure Type for index: ' +str(i))
    return outlist

def add_bonds(view_ats, 
              mol, 
              labelsize=12,
              distvisradius=0.3, 
              distcolor='black',
              distskin=0.3,
              distopacity=0.85,
              vis_distances=None,
              distradius=None,
              distlabelposit=1.0,
              viewer=None):
    """Add bonds to visualization displayer?

    Parameters
    ----------
    view_ats : py3dmol viewer
        py3dmol viewer
    mol : architector.io_molecule.Molecule
        molecule
    labelsize, int
        Size of labels (default 12)
    vis_distances : int/bool/list(int)/str/None,
        Add visualization of distances? Calculate from given indices or from metal.
        e.g. vis_distances=True will add arrows and labels from the metal centers to nearby atoms.
        vis_distances='metals' will do the same
        vis_distances=0 will add arrows and distance labels from the atom 0 to nearby atoms.
        vis_distances=[0,1] will add arrows and distances labesl from both atoms 0 and 1 to nearby atoms.
    distvisradius : float,
        radius of drawn distance vectors, by default 0.3
    distcolor : str,
        color of drawn distance vectors, by default 'black'
    distopacity: float,
        opacity (from 0(transparent) to 1(no transparent) for drawn distance vectors, by default 0.85
    distskin : float,
        skin around given atom to flag "nearby" neighbors, by default 0.3
    distradius : float,
        Radius around a given atom to flag "nearby" neighbors, by default None.
    distlabelposit : float,
        Fraction of the distance (towards the ending atom) that the distance label should be placed, by default 1.0 
    viewer : None,
        which viewer to add the arrows to, by default None
    """
    if vis_distances is not None:
        bondsdf = mol.get_lig_dists(calc_nonbonded_dists=True,
                                    skin=distskin,
                                    ref_ind=vis_distances,
                                    radius=distradius)
        visited = list()
        count = 0
        for i,row in bondsdf.iterrows():
            # Allow for multiple different colors of interatomic distances.
            if (row['atom_pair'][0] in visited) and (isinstance(distcolor,(list,np.ndarray))):
                tcolor = distcolor[visited.index(row['atom_pair'][0])]
            elif (isinstance(distcolor,(list,np.ndarray))):
                tcolor = distcolor[count]
                visited.append(row['atom_pair'][0])
                count += 1
            else:
                tcolor = distcolor
            starting = mol.ase_atoms.get_positions()[row['atom_pair'][0]] # Should be metal.
            ending = mol.ase_atoms.get_positions()[row['atom_pair'][1]]
            sx= starting[0]
            sy= starting[1]
            sz= starting[2]
            ex= ending[0]
            ey= ending[1]
            ez= ending[2]
            dxyz = np.array([ex-sx,ey-sy,ez-sz])
            vector = {'start': {'x':sx, 'y':sy, 'z':sz}, 'end': {'x':ex, 'y':ey, 'z':ez},
                'radius':distvisradius,'color':tcolor,'opacity':distopacity}
            lposit = starting + distlabelposit * dxyz
            if viewer is None:
                view_ats.addArrow(vector)
                view_ats.addLabel("{0:.2f}".format(row['distance']), {'position':{'x':'{}'.format(lposit[0]),
                        'y':'{}'.format(lposit[1]),'z':'{}'.format(lposit[2])},
                        'backgroundColor':"'black'",'backgroundOpacity':'0.3',
                        'fontOpacity':'1', 'fontSize':'{}'.format(labelsize),
                        'fontColor':"white",'inFront':'true'})
            else:
                view_ats.addArrow(vector,viewer=viewer)
                view_ats.addLabel("{0:.2f}".format(row['distance']), {'position':{'x':'{}'.format(lposit[0]),
                        'y':'{}'.format(lposit[1]),'z':'{}'.format(lposit[2])},
                        'backgroundColor':"'black'",'backgroundOpacity':'0.3',
                        'fontOpacity':'1', 'fontSize':'{}'.format(labelsize),
                        'fontColor':"white",'inFront':'true'},viewer=viewer)


            
def view_structures(structures, w=200, h=200, columns=4, representation='ball_stick', labelsize=12,
                 labels=False, labelinds=None, vector=None, sphere_scale=0.3, stick_scale=0.25,
                 metal_scale=0.75, modes=None, trajectory=False, interval=200, vis_distances=None,
                 distvisradius=0.3, distcolor='black', distopacity=0.85, distskin=0.3, distradius=None,
                 distlabelposit=1.0):
    """view_structures
    Jupyter-notebook-based visualization of molecular structures.

    Structures can be anything from a file (.xyz, .mol2, .rxyz), ase Atoms, list (or array) of files, 
    or list/array-like of structure strings, or list/array of ase Atoms.

    Examples:
    view_structures(ase.atoms.Atoms) gives a single viewer with the given structure.
    view_structures('thing.xyz') gives a single viewer with the given structure.
    view_structures(list_of_xyz_strings) gives a grid_view with 4 columns of all xyz strings passed.
    view_structures(list_of_mol2strings) gives a grid_view with 4 columns of all mol2 strings. Will maintain bond orders specified.
    view_structures(pd.Series of mol2strings, labels=pd.OtherSeries of Strings) gives a grid_view with 4 columns with labels superimposed.
    view_structures(mol2string,labelinds=True) gives a single viewer with index of all atoms superimposed as labels.
    view_structures(mol2string,labelinds=list_of_strings) gives a single viewer with the strings put on the atoms with matching indices.
    view_structures(metal_complex_mol2string,
                    vis_distances=True) Will visualize metal-ligand bond distances on the inset images
    view_structures(ase.atoms.Atoms,modes=[vibrational_mode_array]) gives a single viewer with vibrational mode array superimposed
    view_structures([ase.atoms.Atoms]*n,modes=[vibrational_mode_array1,vibrational_mmode_array2....]]) 
    gives a grid viewer with all vibrational modes visualized
    view_structures([trajectory_of_xyzs],trajectory=True) gives a single viewer with the trajectory visualized.

    There is much more functionality to play with.
    Most of what I end up changing is w (width) and h (height) in pixels, and columns (int).
    These specifiy the size of each viewer panel, and number of columns, respecitively. 
    Parameters
    ----------
    structures : str,list,array-like
        structures to visualize
    w : int, optional
        width of the frame or frames (tiled views) to visualize in pixels, by default 200
    h : int, optional
        height of the frame or frames (tiled views) to visualize in pixels, by default 200
    columns : int, optional
        number of columns to split multiple structures into, by default 4
    representation : str, optional
        What molecular representation ('stick','sphere'), by default 'ball_stick'
    labelsize : int, optional
        Fontsize for overlaid text labels, by default 12
    labels : bool, optional
        List or list of strings of labels to add to structures, by default False
    labelinds : bool, list, optional
        Whether to label the indices in each structure, if array passed will use array on matching atom indices, by default None
    vector : dict, optional
        Add arrow? e.g. vector = {'start': {'x':-10.0, 'y':0.0, 'z':0.0}, 'end': {'x':-10.0, 'y':0.0, 'z':10.0},
              'radius':2,'color':'red'}, by default None
    sphere_scale : float, optional
        How large should the spheres be?, by default 0.3
    stick_scale : float, optional
        How large should the sticks be?, by default 0.25
    metal_scale : float, optional
        How large should the metals be?, by default 0.75
    modes : bool/list(np.ndarray), optional
        vibrational modes to animate on structure, by default None
    trajectory : bool, optional
        Whether to view as a trajectory animation (e.g. relaxation or MD), by default False
    interval : int, optional
        How long the trajectory animation should be (speed) incease to move slower, decrease to speed up, by default 200
    vis_distances : int/bool/list(int)/str/None,
        Add visualization of distances? Calculate from given indices or from metal.
        e.g. vis_distances=True will add arrows and labels from the metal centers to nearby atoms.
        vis_distances='metals' will do the same
        vis_distances=0 will add arrows and distance labels from the atom 0 to nearby atoms.
        vis_distances=[0,1] will add arrows and distances labesl from both atoms 0 and 1 to nearby atoms.
    distvisradius : float,
        radius of drawn distance vectors, by default 0.3
    distcolor : str,
        color of drawn distance vectors, by default 'black'
    distopacity: float,
        opacity (from 0(transparent) to 1(no transparent) for drawn distance vectors, by default 0.85
    distskin : float,
        "Skin" on top of sum of cov radii around given atom to flag "nearby" neighbors, by default 0.3
    distradius : float,
        Radius around a given atom to flag "nearby" neighbors, by default None.
    distlabelposit : float,
        Fraction of the distance (towards the ending atom) that the distance label should be placed, by default 1.0 
    """
    mols = type_convert(structures)
    if len(mols) == 1:
        view_ats = py3Dmol.view(width=w,height=h)
        mol = mols[0]
        if isinstance(labels,str):
            label = labels
        elif isinstance(labels,list):
            label = labels[0]
        elif isinstance(labels,bool):
            if labels:
                label = mol.ase_atoms.get_chemical_formula()
            else:
                label = False
        metal_ind = [i for i,x in enumerate(mol.ase_atoms) if (x.symbol in io_ptable.all_metals)]
        if len(metal_ind) > 0 : # Take advantage of empty list
            label_posits = mol.ase_atoms.positions[metal_ind].flatten()
        else:
            label_posits = mol.ase_atoms.get_center_of_mass().flatten()  # Put it at the geometric center of the molecule.
        if modes is not None:
            syms = mol.ase_atoms.get_chemical_symbols()
            atom_coords = mol.ase_atoms.get_positions()
            xyz =f"{len(atom_coords)}\n\n"
            mode_coords = modes[0]
            for i,sym in enumerate(syms):
                xyz+=f"{sym} {atom_coords[i][0]} {atom_coords[i][1]} {atom_coords[i][2]} {mode_coords[i][0]} {mode_coords[i][1]} {mode_coords[i][2]} \n"
        else:
            coords = mol.write_mol2('tmp.mol2', writestring=True)
        if representation == 'ball_stick':
            if modes is not None:
                view_ats.addModel(xyz,'xyz',{'vibrate': {'frames':10,'amplitude':1}})
                view_ats.animate({'loop': 'backAndForth'})
            else:
                view_ats.addModel(coords.replace('un','1'),'mol2') # Add the molecule
            view_ats.addStyle({'sphere':{'colorscheme':'Jmol','scale':sphere_scale}}) 
            msyms = [mol.ase_atoms.get_chemical_symbols()[x] for x in metal_ind]
            for ms in set(msyms):
                view_ats.setStyle({'elem':ms},{'sphere':{'colorscheme':'Jmol','scale':metal_scale}})
            view_ats.addStyle({'stick':{'colorscheme':'Jmol','radius':stick_scale}}) 
            if label:
                view_ats.addLabel("{}".format(label), {'position':{'x':'{}'.format(label_posits[0]),
                    'y':'{}'.format(label_posits[1]),'z':'{}'.format(label_posits[2])},
                    'backgroundColor':"'black'",'backgroundOpacity':'0.3',
                    'fontOpacity':'1', 'fontSize':'{}'.format(labelsize),
                    'fontColor':"white",'inFront':'true'})
        else:
            if modes is not None:
                view_ats.addModel(xyz,'xyz',{'vibrate':{'frames':10,'amplitude':1}})
                view_ats.animate({'loop':'backAndForth'})
            else:
                view_ats.addModel(coords.replace('un','1'),'mol2') # Add the molecule
            if representation == 'stick':
                view_ats.setStyle({representation:{'colorscheme':'Jmol','radius':stick_scale}})
            elif representation == 'sphere':
                view_ats.setStyle({representation:{'colorscheme':'Jmol','scale':sphere_scale}})
            else:
                view_ats.setStyle({representation:{'colorscheme':'Jmol'}})
            if label:
                view_ats.addLabel("{}".format(label), {'position':{'x':'{}'.format(label_posits[0]),
                    'y':'{}'.format(label_posits[1]),'z':'{}'.format(label_posits[2])},
                    'backgroundColor':"'black'",'backgroundOpacity':'0.3',
                    'fontOpacity':'1', 'fontSize':'{}'.format(labelsize),
                    'fontColor':"white",'inFront':'true'})
        if labelinds is not None:
            if isinstance(labelinds,list):
                inds = labelinds
            else:
                inds = [x for x in range(len(mol.ase_atoms))]
            for p,i in enumerate(inds):
                atom_posit = mol.ase_atoms.positions[p]
                if i is not None:
                    view_ats.addLabel("{}".format(i), {'position':{'x':'{}'.format(atom_posit[0]),
                    'y':'{}'.format(atom_posit[1]),'z':'{}'.format(atom_posit[2])},
                    'backgroundColor':"'black'",'backgroundOpacity':'0.4',
                    'fontOpacity':'1', 'fontSize':'{}'.format(labelsize),
                    'fontColor':"white",'inFront':'true'})
        if vector:
            view_ats.addArrow(vector)
        add_bonds(view_ats, mol, 
                  labelsize=labelsize,
                  distvisradius=distvisradius,
                  distcolor=distcolor,
                  distskin=distskin,
                  distopacity=distopacity,
                  distradius=distradius,
                  distlabelposit=distlabelposit,
                  vis_distances=vis_distances)
        view_ats.zoomTo()
        view_ats.show()
    elif (len(mols) < 50) and (not trajectory):
        rows = int(m.ceil(float(len(mols))/columns))
        w = w*columns
        h = h*rows 
        # Initialize Layout
        view_ats = py3Dmol.view(width=w,height=h,linked=False,viewergrid=(rows,columns))
        # Check for labels and populate
        if isinstance(labels,bool):
            if labels:
                label = [x.ase_atoms.get_chemical_formula() for x in mols]
            else:
                label = []
        elif isinstance(labels,list) or isinstance(labels,np.ndarray):
            if (len(labels) != len(mols)):
                print('Wrong amount of labels passed, defaulting to chemical formulas.')
                label = [x.ase_atoms.get_chemical_formula() for x in mols]
            else: # Force them all to be strings. 
                label = [str(x) for x in labels]
        else:
            raise ValueError('What sort of labels are wanting? Not recognized.')
        x,y = 0,0 # Subframe position
        for k,mol in enumerate(mols):
            metal_inds = [i for i,x in enumerate(mol.ase_atoms) if (x.symbol in io_ptable.all_metals)]
            if len(metal_inds) > 0 : # Take advantage of empty list
                label_posits = mol.ase_atoms.positions[metal_inds[0]].flatten()
            else:
                label_posits = mol.ase_atoms.get_center_of_mass().flatten()  # Put it at the geometric center of the molecule.
            if modes is not None:
                atom_coords = mol.ase_atoms.get_positions()
                syms = mol.ase_atoms.get_chemical_symbols()
                xyz =f"{len(atom_coords)}\n\n"
                mode_coords = modes[k]
                for i,sym in enumerate(syms):
                    xyz+=f"{sym} {atom_coords[i][0]} {atom_coords[i][1]} {atom_coords[i][2]} {mode_coords[i][0]} {mode_coords[i][1]} {mode_coords[i][2]} \n"
            else:
                coords = mol.write_mol2('tmp.mol2', writestring=True)
            if representation == 'ball_stick':
                if modes is not None:
                    view_ats.addModel(xyz,'xyz',{'vibrate':{'frames':10,'amplitude':1}},viewer=(x,y))
                    view_ats.animate({'loop':'backAndForth'},viewer=(x,y))
                else:
                    view_ats.addModel(coords.replace('un','1'),'mol2',viewer=(x,y)) # Add the molecule
                view_ats.addStyle({'sphere':{'colorscheme':'Jmol','scale':sphere_scale}},viewer=(x,y)) 
                msyms = [mol.ase_atoms.get_chemical_symbols()[x] for x in metal_inds]
                for ms in set(msyms):
                    view_ats.setStyle({'elem':ms},{'sphere':{'colorscheme':'Jmol','scale':metal_scale}},viewer=(x,y))
                view_ats.addStyle({'stick':{'colorscheme':'Jmol','radius':stick_scale}},viewer=(x,y)) 
                if len(label) > 0:
                    view_ats.addLabel("{}".format(label[k]), {'position':{'x':'{}'.format(label_posits[0]),
                        'y':'{}'.format(label_posits[1]),'z':'{}'.format(label_posits[2])},
                        'backgroundColor':"'black'",'backgroundOpacity':'0.5',
                        'fontOpacity':'1','fontSize':'{}'.format(labelsize),
                        'fontColor':"white",'inFront':'true',}, viewer=(x,y))
                if labelinds is not None:
                    if isinstance(labelinds,list):
                        inds = labelinds[k]
                    else:
                        inds = [x for x in range(len(mol.ase_atoms))]
                    for p,j in enumerate(inds):
                        atom_posit = mol.ase_atoms.positions[p]
                        if j is not None:
                            view_ats.addLabel("{}".format(j), {'position':{'x':'{}'.format(atom_posit[0]),
                            'y':'{}'.format(atom_posit[1]),'z':'{}'.format(atom_posit[2])},
                            'backgroundColor':"'black'",'backgroundOpacity':'0.4',
                            'fontOpacity':'1', 'fontSize':'{}'.format(int(labelsize)),
                            'fontColor':"white", 'inFront':'true'}, viewer=(x,y))
            else:
                if modes is not None:
                    view_ats.addModel(xyz,'xyz',{'vibrate':{'frames':10,'amplitude':1}},viewer=(x,y))
                    view_ats.animate({'loop':'backAndForth'},viewer=(x,y))
                else:
                    view_ats.addModel(coords.replace('un','1'),'mol2',viewer=(x,y)) # Add the molecule
                if representation == 'stick':
                    view_ats.setStyle({representation:{'colorscheme':'Jmol','radius':stick_scale}},viewer=(x,y))
                elif representation == 'sphere':
                    view_ats.setStyle({representation:{'colorscheme':'Jmol','scale':sphere_scale}},viewer=(x,y))
                else:
                    view_ats.setStyle({representation:{'colorscheme':'Jmol'}},viewer=(x,y))
                if len(label) > 0:
                    view_ats.addLabel("{}".format(label[k]), {'position':{'x':'{}'.format(label_posits[0]),
                        'y':'{}'.format(label_posits[1]),'z':'{}'.format(label_posits[2])},
                        'backgroundColor':"'black'",'backgroundOpacity':'0.5',
                        'fontOpacity':'1','fontSize':'{}'.format(labelsize),
                        'fontColor':"white",'inFront':'true',}, viewer=(x,y))
                if labelinds is not None:
                    if isinstance(labelinds,list):
                        inds = labelinds[k]
                    else:
                        inds = [x for x in range(len(mol.ase_atoms))]
                    for p,j in enumerate(inds):
                        atom_posit = mol.ase_atoms.positions[p]
                        if j is not None:
                            view_ats.addLabel("{}".format(j), {'position':{'x':'{}'.format(atom_posit[0]),
                            'y':'{}'.format(atom_posit[1]),'z':'{}'.format(atom_posit[2])},
                            'backgroundColor':"'black'",'backgroundOpacity':'0.4',
                            'fontOpacity':'1', 'fontSize':'{}'.format(int(labelsize)),
                            'fontColor':"white", 'inFront':'true'}, viewer=(x,y))
            if vector:
                view_ats.addArrow(vector, viewer=(x,y))
            add_bonds(view_ats, mol, 
                      distvisradius=distvisradius,
                      distcolor=distcolor,
                      distskin=distskin,
                      distopacity=distopacity,
                      distradius=distradius,
                      distlabelposit=distlabelposit,
                      labelsize=labelsize, vis_distances=vis_distances, viewer=(x,y))
            view_ats.zoomTo(viewer=(x,y))
            if y+1 < columns: # Fill in columns
                y+=1
            else:
                x+=1
                y=0
        view_ats.show()
    elif trajectory: # Animate a relaxation.
        view_ats = py3Dmol.view(width=w,height=h)
        metal_inds = [i for i,x in enumerate(mols[0].ase_atoms) if (x.symbol in io_ptable.all_metals)]
        xyz = ""
        for k,mol in enumerate(mols):
            atom_coords = mol.ase_atoms.get_positions()
            syms = mol.ase_atoms.get_chemical_symbols()
            xyz += f"{len(atom_coords)}\n\n"
            for i,sym in enumerate(syms):
                xyz += f"{sym} {atom_coords[i][0]} {atom_coords[i][1]} {atom_coords[i][2]} \n"
        view_ats.addModelsAsFrames(xyz,'xyz')
        if representation == 'ball_stick':
            view_ats.addStyle({'sphere':{'colorscheme':'Jmol','scale':sphere_scale}}) 
            msyms = [mol.ase_atoms.get_chemical_symbols()[x] for x in metal_inds]
            for ms in set(msyms):
                view_ats.setStyle({'elem':ms},{'sphere':{'colorscheme':'Jmol','scale':metal_scale}})
            view_ats.addStyle({'stick':{'colorscheme':'Jmol','radius':stick_scale}}) 
        else:
            if representation == 'stick':
                view_ats.setStyle({representation:{'colorscheme':'Jmol','radius':stick_scale}})
            elif representation == 'sphere':
                view_ats.setStyle({representation:{'colorscheme':'Jmol','scale':sphere_scale}})
            else:
                view_ats.setStyle({representation:{'colorscheme':'Jmol'}})
        if vector:
            view_ats.addArrow(vector)
        add_bonds(view_ats, mol, 
                  distvisradius=distvisradius,
                  distcolor=distcolor,
                  distskin=distskin,
                  distopacity=distopacity,
                  distradius=distradius,
                  distlabelposit=distlabelposit,
                  labelsize=labelsize, vis_distances=vis_distances)
        view_ats.zoomTo()
        view_ats.animate({'interval':interval,'loop':'forward'}) # Infinite repetition
        view_ats.show()
    else: 
        raise ValueError('Warning. Passing this many structures WILL cause your kernel to crash.')