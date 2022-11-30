"""
Py3Dmol install: (works in both Python2/3 conda environments)
conda install -c conda-forge py3dmol 
Some Documentation: https://pypi.org/project/py3Dmol/
3DMol.js backend: http://3dmol.csb.pitt.edu/index.html

Visualization routine for handling jupyter notebook use of architector.

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
        structures = [convert_io_molecule(structures)]
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
                
            
def view_structures(structures,w=200,h=200,columns=4,representation='stick',labelsize=12,
                 labels=False, labelinds=False, vector=None):
    """
    py3Dmol view atoms object(s)
    xyz_names = xyz files that will be rendered in a tiled format in jupyter (list,str)
    w = width of frame (or subframes) in pixels (int)
    h = height of frame (or subframes) in pixels (int)
    cols = number of columns in subframe (int)
    representation = how the molecule will be viewed (str)
    labelsize = size of the data label (in Points) (int)
    labels = turn labels on/off (bool)
    labelinds = whether to add the indices as text objects to the visualized molecule.
    vector = {'start': {'x':-10.0, 'y':0.0, 'z':0.0}, 'end': {'x':-10.0, 'y':0.0, 'z':10.0},
              'radius':2,'color':'red'}
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
        coords = mol.write_mol2('tmp.mol2', writestring=True)
        view_ats.addModel(coords.replace('un','1'),'mol2') # Add the molecule
        view_ats.setStyle({representation:{'colorscheme':'Jmol'}}) 
        if label:
            view_ats.addLabel("{}".format(label), {'position':{'x':'{}'.format(label_posits[0]),
                  'y':'{}'.format(label_posits[1]),'z':'{}'.format(label_posits[2])},
                  'backgroundColor':"'black'",'backgroundOpacity':'0.3',
                  'fontOpacity':'1', 'fontSize':'{}'.format(labelsize),
                  'fontColor':"white",'inFront':'true'})
        if labelinds:
            inds = [x for x in range(len(mol.ase_atoms))]
            for i in inds:
                atom_posit = mol.ase_atoms.positions[i]
                view_ats.addLabel("{}".format(i), {'position':{'x':'{}'.format(atom_posit[0]),
                  'y':'{}'.format(atom_posit[1]),'z':'{}'.format(atom_posit[2])},
                  'backgroundColor':"'black'",'backgroundOpacity':'0.4',
                  'fontOpacity':'1', 'fontSize':'{}'.format(labelsize),
                  'fontColor':"white",'inFront':'true'})
        if vector:
            view_ats.addArrow(vector)
        view_ats.zoomTo()
        view_ats.show()
    elif len(mols) < 50:
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
        for i,mol in enumerate(mols):
            metal_inds = [i for i,x in enumerate(mol.ase_atoms) if (x.symbol in io_ptable.all_metals)]
            if len(metal_inds) > 0 : # Take advantage of empty list
                label_posits = mol.ase_atoms.positions[metal_inds[0]].flatten()
            else:
                label_posits = mol.ase_atoms.get_center_of_mass().flatten()  # Put it at the geometric center of the molecule.
            coords = mol.write_mol2('tmp.mol2', writestring=True)
            view_ats.addModel(coords.replace('un','1'),'mol2',viewer=(x,y))
            view_ats.setStyle({representation:{'colorscheme':'Jmol'}},viewer=(x,y))
            if len(label) > 0:
                view_ats.addLabel("{}".format(label[i]), {'position':{'x':'{}'.format(label_posits[0]),
                    'y':'{}'.format(label_posits[1]),'z':'{}'.format(label_posits[2])},
                    'backgroundColor':"'black'",'backgroundOpacity':'0.5',
                    'fontOpacity':'1','fontSize':'{}'.format(labelsize),
                    'fontColor':"white",'inFront':'true',}, viewer=(x,y))
            if labelinds:
                inds = [x for x in range(len(mol.ase_atoms))]
                for j in inds:
                    atom_posit = mol.ase_atoms.positions[j]
                    view_ats.addLabel("{}".format(j), {'position':{'x':'{}'.format(atom_posit[0]),
                    'y':'{}'.format(atom_posit[1]),'z':'{}'.format(atom_posit[2])},
                    'backgroundColor':"'black'",'backgroundOpacity':'0.4',
                    'fontOpacity':'1', 'fontSize':'{}'.format(int(labelsize)),
                    'fontColor':"white", 'inFront':'true'}, viewer=(x,y))
            if vector:
                view_ats.addArrow(vector)
            view_ats.zoomTo(viewer=(x,y))
            if y+1 < columns: # Fill in columns
                y+=1
            else:
                x+=1
                y=0
        view_ats.show()
    else: 
        raise ValueError('Warning. Passing this many structures WILL cause your kernel to crash.')
