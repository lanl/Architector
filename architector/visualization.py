"""
Py3Dmol install: (works in both Python2/3 conda environments)
conda install -c conda-forge py3dmol
Some Documentation: https://pypi.org/project/py3Dmol/
3DMol.js backend: http://3dmol.csb.pitt.edu/index.html

pymol install:
conda install -c conda-forge pymol-open-source

Visualization routine for handling jupyter notebook use of architector.

Normal modes code adapted from: https://github.com/duerrsimon/normal-mode-jupyter

Developed by Michael Taylor

pymol default rendering style by Thomas Summers.
"""

import os
import sys
import shutil
import re
from contextlib import contextmanager
import math as m
import pathlib
import numpy as np
import subprocess
import ase

import py3Dmol

import architector
from architector.io_molecule import convert_io_molecule
import architector.io_ptable as io_ptable
from architector.io_align_mol import mirror_permute_align_rmsd
from ase.io import read


def type_convert(structures, hydrogens=True):
    """Handle multiple types of structures passed. List of xyz, mol2 files,
    or list of xyz, mol2strings.

    Parameters
    ----------
    structures : list
        Structures you want visualized: can either be a list or individual:
        mol2 strings, mol2 files, xyz strings, xyz files, or mol3D objects
    hydrogens : bool, optional
        Have hydrogens present?, default True
    """
    outlist = []
    if isinstance(structures, str):
        if (".traj" in structures) or (".gz" in structures):
            structures = read(structures, index=":")
        elif ".xyz" in structures:
            things = read(structures, index=":")
            if len(things) > 1:
                structures = things
        if isinstance(structures, list) and (len(structures) > 1):
            outlist = []
            for i, x in enumerate(structures):
                try:
                    mol = convert_io_molecule(x)
                    outlist.append(mol)
                except:
                    raise ValueError(
                        "Not Recognized Structure Type for index: " + str(i)
                    )
        elif isinstance(structures, list):
            structures = [convert_io_molecule(structures[0])]
        else:
            out = convert_io_molecule(structures)
            if isinstance(out, (np.ndarray, list)):  # Read in traj.
                structures = out
            else:
                structures = [out]
    elif isinstance(
        structures, (ase.atoms.Atoms, architector.io_molecule.Molecule)
    ):
        structures = [convert_io_molecule(structures)]
    elif isinstance(structures, dict):
        try:
            structures = [val["mol2string"] for key, val in structures.items()]
        except:
            raise ValueError(
                "Not recognized type for this dictionary to visualize!"
            )
    else:  # Convert other array-like arguments to a list.
        structures = list(structures)
    if isinstance(structures, list):
        for i, x in enumerate(structures):
            try:
                mol = convert_io_molecule(x)
                outlist.append(mol)
            except:
                raise ValueError(
                    "Not Recognized Structure Type for index: " + str(i)
                )
    if not hydrogens:
        for mol in outlist:
            mol.remove_hydrogens()
    return outlist


def add_bonds(
    view_ats,
    mol,
    labelsize=12,
    distvisradius=0.3,
    distatompairs=None,
    distatomtype_pairs=None,
    distcolor="black",
    distskin=0.3,
    distopacity=0.85,
    vis_distances=None,
    distradius=None,
    distlabelposit=1.0,
    viewer=None,
):
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
    distatompairs : list,
        Atom pairs to view distances.
        e.g. [[0,1],[1,2]] will show only distances between a0 and a1, and a1 and a2, by default None
    distatomtype_pairs: list,
        Atom type pairs to view distances.
        Give multiple sets of atom types to give distnaces [['Fe','O]] should give all fe-o distances...
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
        bondsdf = mol.get_dists(
            calc_nonbonded_dists=True,
            skin=distskin,
            ref_ind=vis_distances,
            radius=distradius,
            atom_pairs=distatompairs,
            atom_type_pairs=distatomtype_pairs,
        )
        visited = list()
        count = 0
        for i, row in bondsdf.iterrows():
            # Allow for multiple different colors of interatomic distances.
            if (row["atom_pair"][0] in visited) and (
                hasattr(distcolor, "__len__")
            ):
                tcolor = distcolor[visited.index(row["atom_pair"][0])]
            elif hasattr(distcolor, "__len__"):
                tcolor = distcolor[count]
                visited.append(row["atom_pair"][0])
                count += 1
            else:
                tcolor = distcolor
            starting = mol.ase_atoms.get_positions()[
                row["atom_pair"][0]
            ]  # Should be metal.
            ending = mol.ase_atoms.get_positions()[row["atom_pair"][1]]
            sx = starting[0]
            sy = starting[1]
            sz = starting[2]
            ex = ending[0]
            ey = ending[1]
            ez = ending[2]
            dxyz = np.array([ex - sx, ey - sy, ez - sz])
            vector = {
                "start": {"x": sx, "y": sy, "z": sz},
                "end": {"x": ex, "y": ey, "z": ez},
                "radius": distvisradius,
                "color": tcolor,
                "opacity": distopacity,
            }
            lposit = starting + distlabelposit * dxyz
            if viewer is None:
                view_ats.addArrow(vector)
                view_ats.addLabel(
                    "{0:.2f}".format(row["distance"]),
                    {
                        "position": {
                            "x": "{}".format(lposit[0]),
                            "y": "{}".format(lposit[1]),
                            "z": "{}".format(lposit[2]),
                        },
                        "backgroundColor": "'black'",
                        "backgroundOpacity": "0.3",
                        "fontOpacity": "1",
                        "fontSize": "{}".format(labelsize),
                        "fontColor": "white",
                        "inFront": "true",
                    },
                )
            else:
                view_ats.addArrow(vector, viewer=viewer)
                view_ats.addLabel(
                    "{0:.2f}".format(row["distance"]),
                    {
                        "position": {
                            "x": "{}".format(lposit[0]),
                            "y": "{}".format(lposit[1]),
                            "z": "{}".format(lposit[2]),
                        },
                        "backgroundColor": "'black'",
                        "backgroundOpacity": "0.3",
                        "fontOpacity": "1",
                        "fontSize": "{}".format(labelsize),
                        "fontColor": "white",
                        "inFront": "true",
                    },
                    viewer=viewer,
                )


def view_structures(
    structures,
    w=200,
    h=200,
    columns=4,
    representation="ball_stick",
    labelsize=12,
    labels=False,
    labelinds=None,
    labelatoms=False,
    vector=None,
    sphere_scale=0.3,
    stick_scale=0.25,
    metal_scale=0.75,
    modes=None,
    trajectory=False,
    interval=200,
    vis_distances=None,
    distvisradius=0.3,
    distcolor="black",
    distopacity=0.85,
    distskin=0.3,
    distradius=None,
    distlabelposit=1.0,
    distatompairs=None,
    distatomtype_pairs=None,
    stack=False,
    stack_align=True,
    hydrogens=True,
    background_color="white",
    render_pymol=False,
    pmpath="./pymol_renders",
    pymol_w=600,
    pymol_h=600,
    pymol_dpi=300,
    pymol_light_count=1,
    pymol_metal_scale=0.45,
    pymol_stick_scale=0.1,
    pymol_h_scale=0.13,
    pymol_other_scale=0.2,
    pymol_shiny_metal=50,
    pymol_shiny_other=40,
    pymol_reflect_metal=0.2,
    pymol_reflect_other=0.1,
    pymol_transparency_metal=0.0,
    pymol_transparency_other=0.0,
    pymol_molecule_buffer=2.0,
    pymol_stick_option="set stick_color, grey20",
    pymol_tag_indices=[],
    pymol_tag_radii=0.6,
    pymol_tag_color="yellow",
    pymol_tag_transparency=0.55,
    pymol_dont_render=False,
):
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
    These specifiy the size of each viewer panel, and number of columns, respectively.
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
    labelatoms : bool, optional
        Whether to label the atoms by atom-type, by default False. Will supercede labelinds if True.
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
    distatompairs : list,
        Atom pairs to view distances.
        e.g. [[0,1],[1,2]] will show only distances between a0 and a1, and a1 and a2, by default None
    distatomtype_pairs : list,
        Atom type pairs to view distances.
        e.g. [['Fe','O'],['Fe','N']] will give distances between all Fe-O and all Fe-N distances, by default None
    stack : bool,
        Stack all the images in a single viewer, default False.
    stack_align : bool,
        Align all the molecules by rmsd for stacking, default True.
    hydrogens : bool,
        Keep the hydrogens?, default True.
    background_color : str,
        set the background color, default 'white'

    ###### Rendering settings.########
    # Note, if you want to render with pymol recommend to install pymol-open-source from conda!
    render_pymol : bool, optional
        render the images in pymol?, default False
    pmpath : str, optional
        where the pymol molecule sits, by default './pymol_renders'
    pymol_w : int, optional
        width of the frame, by default 600
    pymol_h : int, optional
        height of the frame, by default 600
    pymol_dpi : int, optional
        dpi to render at, by default 300
    pymol_light_count : int, optional
        number of lights to add to pymol scene, default 1
    pymol_metal_scale : float, optional
        metal sphere size, default 0.45
    pymol_stick_scale : float, optional
        stick size, default 0.1
    pymol_h_scale : float, optional
        hydrogen sphere size, default 0.13
    pymol_other_scale : float, optional
        other atom sphere size, default 0.2
    pymol_shiny_metal : float, optional
        shininess of the metal, default 50
    pymol_shiny_other : float, optional
        shininess of non-metals, default 40
    pymol_reflect_metal : float, optional
        reflectivity of the metals, default 0.2
    pymol_reflect_other : float, optional
        reflectivity of non-metals, default 0.1
    pymol_transparency_metal : float, optional
        transparency of the metal, default 0.0
    pymoL_transparency_other : float, optional
        transparency of other atoms, default 0.0
    pymol_molecule_buffer : float, optional
        how much space to add around molecules from the edge of the frame
        increase if molecules are going off of the frame, by default 2.0
    pymol_stick_option : str, optional
        stick options passed to pymol, by default 'set stick_color, grey20'
    pymol_tag_indices : list(int), optional
        which indices to tag, default []
    pymol_tag_radii : float, optional
        radii of the tags, default 0.6
    pymol_tag_color : str, optional
        color of the tag, default "yellow"
    pymol_tag_transparency : float, optional
        transparency of the tag, default 0.55
    pymol_dont_render : bool, optional
        run pymol for rendering?, default False
    """
    mols = type_convert(structures,hydrogens=hydrogens)
    if render_pymol:
        # Check for labels and populate
        if isinstance(labels, bool):
            if labels:
                labels = [
                    x.ase_atoms.get_chemical_formula() + str(i)
                    for i, x in enumerate(mols)
                ]
            else:
                labels = [str(i) for i in range(len(mols))]
        elif hasattr(labels, "__len__"):
            if len(labels) != len(mols):
                print(
                    "Wrong amount of labels passed, defaulting to chemical formulas."
                )
                labels = [
                    x.ase_atoms.get_chemical_formula() + str(i)
                    for i, x in enumerate(mols)
                ]
            else:  # Force them all to be strings.
                labels = [str(x) for x in labels]
        else:
            raise ValueError(
                "What sort of labels are wanting? Not recognized."
            )
        if len(pymol_tag_indices) > 0:
            if len(mols) == 1:  # Only 1 molecule.
                # Test if array. If it is, keep the same. Otherwise make sublist.
                if not isinstance(pymol_tag_indices[0], (list, np.ndarray)):
                    pymol_tag_indices = [pymol_tag_indices]
            else:
                # Test if first element is list/array
                if isinstance(pymol_tag_indices[0], (list, np.ndarray)):
                    # If there's fewer lists than indices.
                    if len(pymol_tag_indices) < len(mols):
                        pymol_tag_indices = pymol_tag_indices + [[]] * (
                            len(mols) - len(pymol_tag_indices)
                        )
                        print(
                            "⚠️ Warning: Multiple molecules being rendered, but fewer tag indices flagged.\n"
                            "I am filling out the rest of the indices with empty lists.\n"
                            "If you want tags to apply to all use the format: \n"
                            'view_structures(["CC","C"], render_pymol=True, \n'
                            "                 pymol_tag_indices=[[0],[0]]\n"
                            "As an example tagging the first carbon in each molecule."
                        )
                    elif len(pymol_tag_indices) != len(mols):
                        print(
                            "⚠️ Warning: Multiple molecules being rendered, but more tag indices than molecules flagged.\n"
                            "I am ignoring extra indices passed.\n"
                            "If you want tags to apply to all use the format: \n"
                            'view_structures(["CC","C"], render_pymol=True, \n'
                            "                 pymol_tag_indices=[[0],[0]]\n"
                            "As an example tagging the first carbon in each molecule."
                        )
                else:  # Assume first element is a number.
                    pymol_tag_indices = [pymol_tag_indices]
                    pymol_tag_indices = pymol_tag_indices + [[]] * (
                        len(mols) - len(pymol_tag_indices)
                    )
                    print(
                        "⚠️ Warning: Multiple molecules being rendered, but nested list of tag_indices not passed.\n"
                        "I am assuming the list should be passed to only the first molecule.\n"
                        "I am filling out the rest of the indices with empty lists.\n"
                        "If you want tags to apply to all use the format: \n"
                        'view_structures(["CC","C"], render_pymol=True, \n'
                        "                 pymol_tag_indices=[[0],[0]]\n"
                        "As an example tagging the first carbon in each molecule."
                    )
        else:
            pymol_tag_indices = [[]] * len(mols)
        for i, mol in enumerate(mols):
            make_pml(
                mol,
                render_name=labels[i],
                pmpath=pmpath,
                w=pymol_w,
                h=pymol_h,
                dpi=pymol_dpi,
                light_count=pymol_light_count,
                molecule_buffer=pymol_molecule_buffer,
                metal_scale=pymol_metal_scale,
                stick_scale=pymol_stick_scale,
                h_scale=pymol_h_scale,
                other_scale=pymol_other_scale,
                shiny_metal=pymol_shiny_metal,
                shiny_other=pymol_shiny_other,
                reflect_metal=pymol_reflect_metal,
                reflect_other=pymol_reflect_other,
                stick_option=pymol_stick_option,
                transparency_other=pymol_transparency_other,
                transparency_metal=pymol_transparency_metal,
                tag_indices=pymol_tag_indices[i],
                tag_color=pymol_tag_color,
                tag_radii=pymol_tag_radii,
                tag_transparency=pymol_tag_transparency,
                render=(not pymol_dont_render),
            )
    elif len(mols) == 1:
        view_ats = py3Dmol.view(width=w, height=h)
        view_ats.setBackgroundColor(background_color)
        mol = mols[0]
        if isinstance(labels, str):
            label = labels
        elif isinstance(labels, list):
            label = labels[0]
        elif isinstance(labels, bool):
            if labels:
                label = mol.ase_atoms.get_chemical_formula()
            else:
                label = False
        metal_ind = [
            i
            for i, x in enumerate(mol.ase_atoms)
            if (x.symbol in io_ptable.all_metals)
        ]
        syms = mol.ase_atoms.get_chemical_symbols()
        if len(metal_ind) > 0:  # Take advantage of empty list
            label_posits = mol.ase_atoms.positions[metal_ind].flatten()
        else:
            label_posits = (
                mol.ase_atoms.get_center_of_mass().flatten()
            )  # Put it at the geometric center of the molecule.
        if modes is not None:
            atom_coords = mol.ase_atoms.get_positions()
            xyz = f"{len(atom_coords)}\n\n"
            mode_coords = modes[0]
            for i, sym in enumerate(syms):
                xyz += f"{sym} {atom_coords[i][0]} {atom_coords[i][1]} {atom_coords[i][2]} {mode_coords[i][0]} {mode_coords[i][1]} {mode_coords[i][2]} \n"
        else:
            coords = mol.write_mol2("tmp.mol2", writestring=True)
        if representation == "ball_stick":
            if modes is not None:
                view_ats.addModel(
                    xyz,
                    "xyz",
                    {
                        "keepH": hydrogens,
                        "vibrate": {"frames": 10, "amplitude": 1},
                    },
                )
                view_ats.animate({"loop": "backAndForth"})
            else:
                view_ats.addModel(
                    coords.replace("un", "1"), "mol2", {"keepH": hydrogens}
                )  # Add the molecule
            view_ats.addStyle(
                {"sphere": {"colorscheme": "Jmol", "scale": sphere_scale}}
            )
            msyms = [
                mol.ase_atoms.get_chemical_symbols()[x] for x in metal_ind
            ]
            for ms in set(msyms):
                view_ats.setStyle(
                    {"elem": ms},
                    {"sphere": {"colorscheme": "Jmol", "scale": metal_scale}},
                )
            view_ats.addStyle(
                {"stick": {"colorscheme": "Jmol", "radius": stick_scale}}
            )
            if label:
                view_ats.addLabel(
                    "{}".format(label),
                    {
                        "position": {
                            "x": "{}".format(label_posits[0]),
                            "y": "{}".format(label_posits[1]),
                            "z": "{}".format(label_posits[2]),
                        },
                        "backgroundColor": "'black'",
                        "backgroundOpacity": "0.3",
                        "fontOpacity": "1",
                        "fontSize": "{}".format(labelsize),
                        "fontColor": "white",
                        "inFront": "true",
                    },
                )
        else:
            if modes is not None:
                view_ats.addModel(
                    xyz,
                    "xyz",
                    {
                        "vibrate": {
                            "keepH": hydrogens,
                            "frames": 10,
                            "amplitude": 1,
                        }
                    },
                )
                view_ats.animate({"loop": "backAndForth"})
            else:
                view_ats.addModel(
                    coords.replace("un", "1"), "mol2", {"keepH": hydrogens}
                )  # Add the molecule
            if representation == "stick":
                view_ats.setStyle(
                    {
                        representation: {
                            "colorscheme": "Jmol",
                            "radius": stick_scale,
                        }
                    }
                )
            elif representation == "sphere":
                view_ats.setStyle(
                    {
                        representation: {
                            "colorscheme": "Jmol",
                            "scale": sphere_scale,
                        }
                    }
                )
            else:
                view_ats.setStyle({representation: {"colorscheme": "Jmol"}})
            if label:
                view_ats.addLabel(
                    "{}".format(label),
                    {
                        "position": {
                            "x": "{}".format(label_posits[0]),
                            "y": "{}".format(label_posits[1]),
                            "z": "{}".format(label_posits[2]),
                        },
                        "backgroundColor": "'black'",
                        "backgroundOpacity": "0.3",
                        "fontOpacity": "1",
                        "fontSize": "{}".format(labelsize),
                        "fontColor": "white",
                        "inFront": "true",
                    },
                )
        if (labelinds is not None) and (not labelatoms):
            if isinstance(labelinds, list):
                inds = labelinds
            else:
                inds = [x for x in range(len(mol.ase_atoms))]
            for p, i in enumerate(inds):
                atom_posit = mol.ase_atoms.positions[p]
                if i is not None:
                    view_ats.addLabel(
                        "{}".format(i),
                        {
                            "position": {
                                "x": "{}".format(atom_posit[0]),
                                "y": "{}".format(atom_posit[1]),
                                "z": "{}".format(atom_posit[2]),
                            },
                            "backgroundColor": "'black'",
                            "backgroundOpacity": "0.4",
                            "fontOpacity": "1",
                            "fontSize": "{}".format(labelsize),
                            "fontColor": "white",
                            "inFront": "true",
                        },
                    )
        elif labelatoms:
            inds = [x for x in range(len(mol.ase_atoms))]
            for p, i in enumerate(inds):
                atom_posit = mol.ase_atoms.positions[p]
                if i is not None:
                    view_ats.addLabel(
                        "{}".format(syms[i]),
                        {
                            "position": {
                                "x": "{}".format(atom_posit[0]),
                                "y": "{}".format(atom_posit[1]),
                                "z": "{}".format(atom_posit[2]),
                            },
                            "backgroundColor": "'black'",
                            "backgroundOpacity": "0.4",
                            "fontOpacity": "1",
                            "fontSize": "{}".format(labelsize),
                            "fontColor": "white",
                            "inFront": "true",
                        },
                    )
        if vector:
            view_ats.addArrow(vector)
        add_bonds(
            view_ats,
            mol,
            labelsize=labelsize,
            distvisradius=distvisradius,
            distcolor=distcolor,
            distskin=distskin,
            distopacity=distopacity,
            distradius=distradius,
            distlabelposit=distlabelposit,
            distatompairs=distatompairs,
            distatomtype_pairs=distatomtype_pairs,
            vis_distances=vis_distances,
        )
        view_ats.zoomTo()
        view_ats.show()
    elif (len(mols) < 50) and (not trajectory) and (not stack):
        rows = int(m.ceil(float(len(mols)) / columns))
        w = w * columns
        h = h * rows
        # Initialize Layout
        view_ats = py3Dmol.view(
            width=w, height=h, linked=False, viewergrid=(rows, columns)
        )
        view_ats.setBackgroundColor(background_color)
        # Check for labels and populate
        if isinstance(labels, bool):
            if labels:
                label = [x.ase_atoms.get_chemical_formula() for x in mols]
            else:
                label = []
        elif hasattr(labels, "__len__"):
            if len(labels) != len(mols):
                print(
                    "Wrong amount of labels passed, defaulting to chemical formulas."
                )
                label = [x.ase_atoms.get_chemical_formula() for x in mols]
            else:  # Force them all to be strings.
                label = [str(x) for x in labels]
        else:
            raise ValueError(
                "What sort of labels are wanting? Not recognized."
            )
        x, y = 0, 0  # Subframe position
        for k, mol in enumerate(mols):
            syms = mol.ase_atoms.get_chemical_symbols()
            metal_inds = [
                i
                for i, x in enumerate(mol.ase_atoms)
                if (x.symbol in io_ptable.all_metals)
            ]
            if len(metal_inds) > 0:  # Take advantage of empty list
                label_posits = mol.ase_atoms.positions[metal_inds[0]].flatten()
            else:
                label_posits = (
                    mol.ase_atoms.get_center_of_mass().flatten()
                )  # Put it at the geometric center of the molecule.
            if modes is not None:
                atom_coords = mol.ase_atoms.get_positions()
                xyz = f"{len(atom_coords)}\n\n"
                mode_coords = modes[k]
                for i, sym in enumerate(syms):
                    xyz += f"{sym} {atom_coords[i][0]} {atom_coords[i][1]} {atom_coords[i][2]} {mode_coords[i][0]} {mode_coords[i][1]} {mode_coords[i][2]} \n"
            else:
                coords = mol.write_mol2("tmp.mol2", writestring=True)
            if representation == "ball_stick":
                if modes is not None:
                    view_ats.addModel(
                        xyz,
                        "xyz",
                        {
                            "vibrate": {"frames": 10, "amplitude": 1},
                            "keepH": hydrogens,
                        },
                        viewer=(x, y),
                    )
                    view_ats.animate({"loop": "backAndForth"}, viewer=(x, y))
                else:
                    view_ats.addModel(
                        coords.replace("un", "1"),
                        "mol2",
                        {"keepH": hydrogens},
                        viewer=(x, y),
                    )  # Add the molecule
                view_ats.addStyle(
                    {"sphere": {"colorscheme": "Jmol", "scale": sphere_scale}},
                    viewer=(x, y),
                )
                msyms = [
                    mol.ase_atoms.get_chemical_symbols()[x] for x in metal_inds
                ]
                for ms in set(msyms):
                    view_ats.setStyle(
                        {"elem": ms},
                        {
                            "sphere": {
                                "colorscheme": "Jmol",
                                "scale": metal_scale,
                            }
                        },
                        viewer=(x, y),
                    )
                view_ats.addStyle(
                    {"stick": {"colorscheme": "Jmol", "radius": stick_scale}},
                    viewer=(x, y),
                )
                if len(label) > 0:
                    view_ats.addLabel(
                        "{}".format(label[k]),
                        {
                            "position": {
                                "x": "{}".format(label_posits[0]),
                                "y": "{}".format(label_posits[1]),
                                "z": "{}".format(label_posits[2]),
                            },
                            "backgroundColor": "'black'",
                            "backgroundOpacity": "0.5",
                            "fontOpacity": "1",
                            "fontSize": "{}".format(labelsize),
                            "fontColor": "white",
                            "inFront": "true",
                        },
                        viewer=(x, y),
                    )
                if (labelinds is not None) and (not labelatoms):
                    if isinstance(labelinds, list):
                        inds = labelinds[k]
                    else:
                        inds = [x for x in range(len(mol.ase_atoms))]
                    for p, j in enumerate(inds):
                        atom_posit = mol.ase_atoms.positions[p]
                        if j is not None:
                            view_ats.addLabel(
                                "{}".format(j),
                                {
                                    "position": {
                                        "x": "{}".format(atom_posit[0]),
                                        "y": "{}".format(atom_posit[1]),
                                        "z": "{}".format(atom_posit[2]),
                                    },
                                    "backgroundColor": "'black'",
                                    "backgroundOpacity": "0.4",
                                    "fontOpacity": "1",
                                    "fontSize": "{}".format(int(labelsize)),
                                    "fontColor": "white",
                                    "inFront": "true",
                                },
                                viewer=(x, y),
                            )
                elif labelatoms:
                    inds = [x for x in range(len(mol.ase_atoms))]
                    for p, i in enumerate(inds):
                        atom_posit = mol.ase_atoms.positions[p]
                        if i is not None:
                            view_ats.addLabel(
                                "{}".format(syms[i]),
                                {
                                    "position": {
                                        "x": "{}".format(atom_posit[0]),
                                        "y": "{}".format(atom_posit[1]),
                                        "z": "{}".format(atom_posit[2]),
                                    },
                                    "backgroundColor": "'black'",
                                    "backgroundOpacity": "0.4",
                                    "fontOpacity": "1",
                                    "fontSize": "{}".format(labelsize),
                                    "fontColor": "white",
                                    "inFront": "true",
                                },
                                viewer=(x, y),
                            )
            else:
                if modes is not None:
                    view_ats.addModel(
                        xyz,
                        "xyz",
                        {
                            "keepH": hydrogens,
                            "vibrate": {"frames": 10, "amplitude": 1},
                        },
                        viewer=(x, y),
                    )
                    view_ats.animate({"loop": "backAndForth"}, viewer=(x, y))
                else:
                    view_ats.addModel(
                        coords.replace("un", "1"),
                        "mol2",
                        {"keepH": hydrogens},
                        viewer=(x, y),
                    )  # Add the molecule
                if representation == "stick":
                    view_ats.setStyle(
                        {
                            representation: {
                                "colorscheme": "Jmol",
                                "radius": stick_scale,
                            }
                        },
                        viewer=(x, y),
                    )
                elif representation == "sphere":
                    view_ats.setStyle(
                        {
                            representation: {
                                "colorscheme": "Jmol",
                                "scale": sphere_scale,
                            }
                        },
                        viewer=(x, y),
                    )
                else:
                    view_ats.setStyle(
                        {representation: {"colorscheme": "Jmol"}},
                        viewer=(x, y),
                    )
                if len(label) > 0:
                    view_ats.addLabel(
                        "{}".format(label[k]),
                        {
                            "position": {
                                "x": "{}".format(label_posits[0]),
                                "y": "{}".format(label_posits[1]),
                                "z": "{}".format(label_posits[2]),
                            },
                            "backgroundColor": "'black'",
                            "backgroundOpacity": "0.5",
                            "fontOpacity": "1",
                            "fontSize": "{}".format(labelsize),
                            "fontColor": "white",
                            "inFront": "true",
                        },
                        viewer=(x, y),
                    )
                if (labelinds is not None) and (not labelatoms):
                    if isinstance(labelinds, list):
                        inds = labelinds[k]
                    else:
                        inds = [x for x in range(len(mol.ase_atoms))]
                    for p, j in enumerate(inds):
                        atom_posit = mol.ase_atoms.positions[p]
                        if j is not None:
                            view_ats.addLabel(
                                "{}".format(j),
                                {
                                    "position": {
                                        "x": "{}".format(atom_posit[0]),
                                        "y": "{}".format(atom_posit[1]),
                                        "z": "{}".format(atom_posit[2]),
                                    },
                                    "backgroundColor": "'black'",
                                    "backgroundOpacity": "0.4",
                                    "fontOpacity": "1",
                                    "fontSize": "{}".format(int(labelsize)),
                                    "fontColor": "white",
                                    "inFront": "true",
                                },
                                viewer=(x, y),
                            )
                elif labelatoms:
                    inds = [x for x in range(len(mol.ase_atoms))]
                    for p, i in enumerate(inds):
                        atom_posit = mol.ase_atoms.positions[p]
                        if i is not None:
                            view_ats.addLabel(
                                "{}".format(syms[i]),
                                {
                                    "position": {
                                        "x": "{}".format(atom_posit[0]),
                                        "y": "{}".format(atom_posit[1]),
                                        "z": "{}".format(atom_posit[2]),
                                    },
                                    "backgroundColor": "'black'",
                                    "backgroundOpacity": "0.4",
                                    "fontOpacity": "1",
                                    "fontSize": "{}".format(labelsize),
                                    "fontColor": "white",
                                    "inFront": "true",
                                },
                            )
            if vector:
                view_ats.addArrow(vector, viewer=(x, y))
            add_bonds(
                view_ats,
                mol,
                distvisradius=distvisradius,
                distcolor=distcolor,
                distskin=distskin,
                distopacity=distopacity,
                distradius=distradius,
                distlabelposit=distlabelposit,
                distatompairs=distatompairs,
                distatomtype_pairs=distatomtype_pairs,
                labelsize=labelsize,
                vis_distances=vis_distances,
                viewer=(x, y),
            )
            view_ats.zoomTo(viewer=(x, y))
            if y + 1 < columns:  # Fill in columns
                y += 1
            else:
                x += 1
                y = 0
        view_ats.show()
    elif trajectory:  # Animate a relaxation.
        view_ats = py3Dmol.view(width=w, height=h)
        view_ats.setBackgroundColor(background_color)
        metal_inds = [
            i
            for i, x in enumerate(mols[0].ase_atoms)
            if (x.symbol in io_ptable.all_metals)
        ]
        xyz = ""
        for k, mol in enumerate(mols):
            atom_coords = mol.ase_atoms.get_positions()
            syms = mol.ase_atoms.get_chemical_symbols()
            xyz += f"{len(atom_coords)}\n\n"
            for i, sym in enumerate(syms):
                xyz += f"{sym} {atom_coords[i][0]} {atom_coords[i][1]} {atom_coords[i][2]} \n"
        view_ats.addModelsAsFrames(xyz, "xyz", {"keepH": hydrogens})
        if representation == "ball_stick":
            view_ats.addStyle(
                {"sphere": {"colorscheme": "Jmol", "scale": sphere_scale}}
            )
            msyms = [
                mol.ase_atoms.get_chemical_symbols()[x] for x in metal_inds
            ]
            for ms in set(msyms):
                view_ats.setStyle(
                    {"elem": ms},
                    {"sphere": {"colorscheme": "Jmol", "scale": metal_scale}},
                )
            view_ats.addStyle(
                {"stick": {"colorscheme": "Jmol", "radius": stick_scale}}
            )
        else:
            if representation == "stick":
                view_ats.setStyle(
                    {
                        representation: {
                            "colorscheme": "Jmol",
                            "radius": stick_scale,
                        }
                    }
                )
            elif representation == "sphere":
                view_ats.setStyle(
                    {
                        representation: {
                            "colorscheme": "Jmol",
                            "scale": sphere_scale,
                        }
                    }
                )
            else:
                view_ats.setStyle({representation: {"colorscheme": "Jmol"}})
        if vector:
            view_ats.addArrow(vector)
        add_bonds(
            view_ats,
            mol,
            distvisradius=distvisradius,
            distcolor=distcolor,
            distskin=distskin,
            distopacity=distopacity,
            distradius=distradius,
            distlabelposit=distlabelposit,
            distatompairs=distatompairs,
            distatomtype_pairs=distatomtype_pairs,
            labelsize=labelsize,
            vis_distances=vis_distances,
        )
        view_ats.zoomTo()
        view_ats.animate(
            {"interval": interval, "loop": "forward"}
        )  # Infinite repetition
        view_ats.show()
    elif stack:
        view_ats = py3Dmol.view(width=w, height=h)
        view_ats.setBackgroundColor(background_color)
        metal_inds = [
            i
            for i, x in enumerate(mols[0].ase_atoms)
            if (x.symbol in io_ptable.all_metals)
        ]
        xyz = ""
        mol0 = mols[0]
        for k, mol in enumerate(mols):
            if stack_align:
                aligned = mirror_permute_align_rmsd(mol0.ase_atoms, mol.ase_atoms)
                newmol = convert_io_molecule(aligned)
                newmol.create_mol_graph()
                mol = newmol
            coords = mol.write_mol2("thing", writestring=True)
            coords = coords.replace("un", "1")
            view_ats.addModel(
                coords.replace("un", "1"), "mol2", {"keepH": hydrogens}
            )  # Add the molecule
        if representation == "ball_stick":
            view_ats.addStyle(
                {"sphere": {"colorscheme": "Jmol", "scale": sphere_scale}}
            )
            msyms = [
                mol.ase_atoms.get_chemical_symbols()[x] for x in metal_inds
            ]
            for ms in set(msyms):
                view_ats.setStyle(
                    {"elem": ms},
                    {"sphere": {"colorscheme": "Jmol", "scale": metal_scale}},
                )
            view_ats.addStyle(
                {"stick": {"colorscheme": "Jmol", "radius": stick_scale}}
            )
        else:
            if representation == "stick":
                view_ats.setStyle(
                    {
                        representation: {
                            "colorscheme": "Jmol",
                            "radius": stick_scale,
                        }
                    }
                )
            elif representation == "sphere":
                view_ats.setStyle(
                    {
                        representation: {
                            "colorscheme": "Jmol",
                            "scale": sphere_scale,
                        }
                    }
                )
            else:
                view_ats.setStyle({representation: {"colorscheme": "Jmol"}})
        if vector:
            view_ats.addArrow(vector)
        add_bonds(
            view_ats,
            mol,
            distvisradius=distvisradius,
            distcolor=distcolor,
            distskin=distskin,
            distopacity=distopacity,
            distradius=distradius,
            distlabelposit=distlabelposit,
            distatompairs=distatompairs,
            distatomtype_pairs=distatomtype_pairs,
            labelsize=labelsize,
            vis_distances=vis_distances,
        )
        view_ats.zoomTo()
        view_ats.show()
    else:
        raise ValueError(
            "Warning. Passing this many structures WILL cause your kernel to crash."
        )


# Many thanks to Thomas Summers for sharing his base style selection.
pymol_python_template = """
#### Note: To re-render run in the pymol_renders directory: ####
### pymol {render_name}.pml # -> Adjust viewport/any settings
#### > ray {size_x}, {size_y}
#### > png {render_name}.png, dpi={dpi}


# Load molecule
load {render_name}.mol2

# Set viewport
viewport {size_x}, {size_y}

# Set display settings
bg_color white
set ray_opaque_background, off
set orthoscopic, 0
set dash_gap, 0
set ray_texture, 0
set antialias, 3
set ambient, 0.5
set spec_count, 5
set shininess, {shiny_other}
set specular, 1
set reflect, {reflect_other}
set stick_radius, {stick_scale}
set transparency, {transparency_other}
set sphere_transparency, {transparency_other}
set dash_gap, 0.01
set light_count, {light_count}
set dash_radius, 0.035
{stick_option}
set sphere_scale, {other_scale}
set sphere_scale, {h_scale}, elem H
# Lanthanides
set sphere_scale, {metal_scale}, elem La+Ce+Pr+Nd+Pm+Sm+Eu+Gd+Tb+Dy+Ho+Er+Tm+Yb+Lu
set shininess, {shiny_metal}, elem La+Ce+Pr+Nd+Pm+Sm+Eu+Gd+Tb+Dy+Ho+Er+Tm+Yb+Lu
set reflect, {reflect_metal}, elem La+Ce+Pr+Nd+Pm+Sm+Eu+Gd+Tb+Dy+Ho+Er+Tm+Yb+Lu
set transparency, {transparency_metal}, elem La+Ce+Pr+Nd+Pm+Sm+Eu+Gd+Tb+Dy+Ho+Er+Tm+Yb+Lu
set sphere_transparency, {transparency_metal}, elem La+Ce+Pr+Nd+Pm+Sm+Eu+Gd+Tb+Dy+Ho+Er+Tm+Yb+Lu
# Actinides
set sphere_scale, {metal_scale}, elem Ac+Th+Pa+U+Np+Pu+Am+Cm+Bk+Cf+Es+Fm+Md+No+Lr
set shininess, {shiny_metal}, elem Ac+Th+Pa+U+Np+Pu+Am+Cm+Bk+Cf+Es+Fm+Md+No+Lr
set reflect, {reflect_metal}, elem Ac+Th+Pa+U+Np+Pu+Am+Cm+Bk+Cf+Es+Fm+Md+No+Lr
set transparency, {transparency_metal}, elem Ac+Th+Pa+U+Np+Pu+Am+Cm+Bk+Cf+Es+Fm+Md+No+Lr
set sphere_transparency, {transparency_metal}, elem Ac+Th+Pa+U+Np+Pu+Am+Cm+Bk+Cf+Es+Fm+Md+No+Lr
# First row
set sphere_scale, {metal_scale}, elem Sc+Ti+V+Cr+Mn+Fe+Co+Ni+Cu+Zn
set shininess, {shiny_metal}, elem Sc+Ti+V+Cr+Mn+Fe+Co+Ni+Cu+Zn
set reflect, {reflect_metal}, elem Sc+Ti+V+Cr+Mn+Fe+Co+Ni+Cu+Zn
set transparency, {transparency_metal}, elem Sc+Ti+V+Cr+Mn+Fe+Co+Ni+Cu+Zn
set sphere_transparency, {transparency_metal}, elem Sc+Ti+V+Cr+Mn+Fe+Co+Ni+Cu+Zn
# Second row
set sphere_scale, {metal_scale}, elem Y+Zr+Nb+Mo+Tc+Ru+Rh+Pd+Ag+Cd
set shininess, {shiny_metal}, elem Y+Zr+Nb+Mo+Tc+Ru+Rh+Pd+Ag+Cd
set reflect, {reflect_metal}, elem Y+Zr+Nb+Mo+Tc+Ru+Rh+Pd+Ag+Cd
set transparency, {transparency_metal}, elem Y+Zr+Nb+Mo+Tc+Ru+Rh+Pd+Ag+Cd
set sphere_transparency, {transparency_metal}, elem Y+Zr+Nb+Mo+Tc+Ru+Rh+Pd+Ag+Cd
# Third row +
set sphere_scale, {metal_scale}, elem Hf+Ta+W+Re+Os+Ir+Pt+Au+Hg+Rf+Db+Sg+Bh+Hs
set shininess, {shiny_metal}, elem Hf+Ta+W+Re+Os+Ir+Pt+Au+Hg+Rf+Db+Sg+Bh+Hs
set reflect, {reflect_metal}, elem Hf+Ta+W+Re+Os+Ir+Pt+Au+Hg+Rf+Db+Sg+Bh+Hs
set transparency, {transparency_metal}, elem Hf+Ta+W+Re+Os+Ir+Pt+Au+Hg+Rf+Db+Sg+Bh+Hs
set sphere_transparency, {transparency_metal}, elem Hf+Ta+W+Re+Os+Ir+Pt+Au+Hg+Rf+Db+Sg+Bh+Hs
# Alakai
set sphere_scale, {metal_scale}, elem Li+Na+K+Rb+Cs+Fr
set shininess, {shiny_metal}, elem Li+Na+K+Rb+Cs+Fr
set reflect, {reflect_metal}, elem Li+Na+K+Rb+Cs+Fr
set transparency, {transparency_metal}, elem Li+Na+K+Rb+Cs+Fr
set sphere_transparency, {transparency_metal}, elem Li+Na+K+Rb+Cs+Fr
# Alakai Earth
set sphere_scale, {metal_scale}, elem Be+Mg+Ca+Sr+Ba+Ra
set shininess, {shiny_metal}, elem Be+Mg+Ca+Sr+Ba+Ra
set reflect, {reflect_metal}, elem Be+Mg+Ca+Sr+Ba+Ra
set transparency, {transparency_metal}, elem Be+Mg+Ca+Sr+Ba+Ra
set sphere_transparency, {transparency_metal}, elem Be+Mg+Ca+Sr+Ba+Ra
# Post transition
set sphere_scale, {metal_scale}, elem Al+Ga+In+Sn+Tl+Pb+Bi+Nh+Fl+Mc+Lv
set shininess, {shiny_metal}, elem Al+Ga+In+Sn+Tl+Pb+Bi+Nh+Fl+Mc+Lv
set reflect, {reflect_metal}, elem Al+Ga+In+Sn+Tl+Pb+Bi+Nh+Fl+Mc+Lv
set transparency, {transparency_metal}, elem Al+Ga+In+Sn+Tl+Pb+Bi+Nh+Fl+Mc+Lv
set sphere_transparency, {transparency_metal}, elem Al+Ga+In+Sn+Tl+Pb+Bi+Nh+Fl+Mc+Lv

{tag_section}

# Set color space (CMYK doesn't affect atom colors, so this is optional)
space cmyk

# Show ball and stick
show sticks
show spheres
hide nonbonded
hide lines
hide labels

# Apply Jmol colors
color 0xFFFFFF, elem H
color 0xD9FFFF, elem He
color 0xCC80FF, elem Li
color 0xC2FF00, elem Be
color 0xFFB5B5, elem B
color 0x909090, elem C
color 0x3050F8, elem N
color 0xFF0D0D, elem O
color 0x90E050, elem F
color 0xB3E3F5, elem Ne
color 0xAB5CF2, elem Na
color 0x8AFF00, elem Mg
color 0xBFA6A6, elem Al
color 0xF0C8A0, elem Si
color 0xFF8000, elem P
color 0xFFFF30, elem S
color 0x1FF01F, elem Cl
color 0x80D1E3, elem Ar
color 0x8F40D4, elem K
color 0x3DFF00, elem Ca
color 0xE6E6E6, elem Sc
color 0xBFC2C7, elem Ti
color 0xA6A6AB, elem V
color 0x8A99C7, elem Cr
color 0x9C7AC7, elem Mn
color 0xE06633, elem Fe
color 0xF090A0, elem Co
color 0x50D050, elem Ni
color 0xC88033, elem Cu
color 0x7D80B0, elem Zn
color 0xC28F8F, elem Ga
color 0x668F8F, elem Ge
color 0xBD80E3, elem As
color 0xFFA100, elem Se
color 0xA62929, elem Br
color 0x5CB8D1, elem Kr
color 0x702EB0, elem Rb
color 0x00FF00, elem Sr
color 0x94FFFF, elem Y
color 0x94E0E0, elem Zr
color 0x73C2C9, elem Nb
color 0x54B5B5, elem Mo
color 0x3B9E9E, elem Tc
color 0x248F8F, elem Ru
color 0x0A7D8C, elem Rh
color 0x006985, elem Pd
color 0xC0C0C0, elem Ag
color 0xFFD98F, elem Cd
color 0xA67573, elem In
color 0x668080, elem Sn
color 0x9E63B5, elem Sb
color 0xD47A00, elem Te
color 0x940094, elem I
color 0x429EB0, elem Xe
color 0x57178F, elem Cs
color 0x00C900, elem Ba
color 0x70D4FF, elem La
color 0xFFFFC7, elem Ce
color 0xD9FFC7, elem Pr
color 0xC7FFC7, elem Nd
color 0xA3FFC7, elem Pm
color 0x8FFFC7, elem Sm
color 0x61FFC7, elem Eu
color 0x45FFC7, elem Gd
color 0x30FFC7, elem Tb
color 0x1FFFC7, elem Dy
color 0x00FF9C, elem Ho
color 0x00E675, elem Er
color 0x00D452, elem Tm
color 0x00BF38, elem Yb
color 0x00AB24, elem Lu
color 0x4DC2FF, elem Hf
color 0x4DA6FF, elem Ta
color 0x2194D6, elem W
color 0x267DAB, elem Re
color 0x266696, elem Os
color 0x175487, elem Ir
color 0xD0D0E0, elem Pt
color 0xFFD123, elem Au
color 0xB8B8D0, elem Hg
color 0xA6544D, elem Tl
color 0x575961, elem Pb
color 0x9E4FB5, elem Bi
color 0xAB5C00, elem Po
color 0x754F45, elem At
color 0x428296, elem Rn
color 0x420066, elem Fr
color 0x007D00, elem Ra
color 0x70ABFA, elem Ac
color 0x00BAFF, elem Th
color 0x00A1FF, elem Pa
color 0x008FFF, elem U
color 0x0080FF, elem Np
color 0x006BFF, elem Pu
color 0x545CF2, elem Am
color 0x785CE3, elem Cm
color 0x8A4FE3, elem Bk
color 0xA136D4, elem Cf
color 0xB31FD4, elem Es
color 0xB31FBA, elem Fm
color 0xB30DA6, elem Md
color 0xBD0D87, elem No
color 0xC70066, elem Lr
color 0xCC0059, elem Rf
color 0xD1004F, elem Db
color 0xD90045, elem Sg
color 0xE00038, elem Bh
color 0xE6002E, elem Hs
color 0xEB0026, elem Mt

# Fit and render
orient
zoom buffer={molecule_buffer}
ray {size_x}, {size_y}
png {render_name}.png, dpi={dpi}

#### Note: To re-render run in the pymol_renders directory: ####
### pymol {render_name}.pml # -> Adjust viewport/any settings
#### > ray {size_x}, {size_y}
#### > png {render_name}.png, dpi={dpi}
"""


@contextmanager
def change_dir(path):
    prev_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_dir)


def display_image_smart(path):
    """Displays an image inline if in Jupyter, or prints a message otherwise."""

    def in_jupyter():
        try:
            from IPython import get_ipython

            shell = get_ipython().__class__.__name__
            return (
                shell == "ZMQInteractiveShell"
            )  # typical Jupyter notebook kernel
        except Exception:
            return False

    if not os.path.exists(path):
        print(f"Image not found: {path}")
        return

    if in_jupyter():
        from IPython.display import Image, display

        display(Image(filename=path))
    else:
        print(f"Not in Jupyter. Image saved at: {path}")
        # Open with default image viewer
        if sys.platform.startswith("darwin"):  # macOS
            os.system(f"open {path}")
        elif os.name == "posix":  # Linux
            os.system(f"xdg-open {path}")


def get_next_version_filename(base_name, ext, directory="."):
    """
    Finds the next versioned filename like 'base_name_vX.ext' in the given directory.
    """
    pattern = re.compile(
        rf"^{re.escape(base_name)}_v(\d+)\.{re.escape(ext)}$", re.IGNORECASE
    )
    version_numbers = []

    for f in os.listdir(directory):
        basename = os.path.basename(f)
        match = pattern.match(basename)
        if match:
            version_numbers.append(int(match.group(1)))

    next_version = max(version_numbers, default=0) + 1
    return f"{base_name}_v{next_version}"


tag_template = """
# Tag atom
pseudoatom {tag_label}, pos={tag_posit}
show spheres, {tag_label}
color {tag_color}, {tag_label}
set sphere_scale, {tag_radii}, {tag_label}
set sphere_transparency, {tag_transparency}, {tag_label}

"""


def make_tag_section(
    molecule,
    tag_indices=[],
    tag_radii=0.6,
    tag_color="yellow",
    tag_transparency=0.55,
):
    """create the tag section

    molecule : architector molecule
        molecule to add tags to.
    tag_indices : list(int), optional
        which indices to tag, default []
    tag_radii : float, optional
        radii of the tags, default 0.6
    tag_color : str, optional
        color of the tag, default "yellow"
    tag_transparency : float, optional
        transparency of the tag, default 0.55
    """
    out = ""
    if len(tag_indices) > 0:
        for i, ind in enumerate(tag_indices):
            tag_label = "tag_at" + str(i)
            tag_posit = (
                "["
                + "{}, {}, {}".format(
                    molecule.ase_atoms.positions[ind][0],
                    molecule.ase_atoms.positions[ind][1],
                    molecule.ase_atoms.positions[ind][2],
                )
                + "]"
            )
            out += tag_template.format(
                tag_label=tag_label,
                tag_posit=tag_posit,
                tag_radii=tag_radii,
                tag_color=tag_color,
                tag_transparency=tag_transparency,
            )
    return out


def make_pml(
    molecule,
    render_name="0",
    pmpath="./pymol_renders",
    w=600,
    h=600,
    dpi=300,
    light_count=1,
    molecule_buffer=2.0,
    metal_scale=0.45,
    stick_scale=0.1,
    h_scale=0.13,
    other_scale=0.2,
    shiny_other=40,
    shiny_metal=50,
    reflect_other=0.1,
    reflect_metal=0.2,
    transparency_other=0,
    transparency_metal=0,
    tag_indices=[],
    tag_radii=0.6,
    tag_color="yellow",
    tag_transparency=0.55,
    stick_option="set stick_color, grey20",
    render=True,
):
    """make_pml -> generates and runs pymol on the molecules and then displays them if it
    is run under jupyter.

    Parameters
    ----------
    molecule : molecule type
        any molecule
    render_name : str, optional
        the name to render the file as, by default '0'
    pmpath : str, optional
        where the pymol molecule sits, by default './pymol_renders'
    w : int, optional
        width of the frame, by default 600
    h : int, optional
        height of the frame, by default 600
    dpi : int, optional
        dpi to render at, by default 300
    light_count : int, optional
        number of lights to add to scene, default 1.
    molecule_buffer : float, optional
        how much space to add around molecules from the edge of the frame
        increase if molecules are going off of the frame, by default 2.0
    metal_scale : float, optional
        metal sphere size, default 0.45
    stick_scale : float, optional
        stick size, default 0.1
    h_scale : float, optional
        hydrogen sphere size, default 0.13
    other_scale : float, optional
        other atom sphere size, default 0.2
    shiny_metal : float, optional
        shininess of the metal, default 50
    shiny_other : float, optional
        shininess of non-metals, default 40
    reflect_metal : float, optional
        reflectivity of the metals, default 0.2
    reflect_other : float, optional
        reflectivity of non-metals, default 0.1
    transparency_metal : float, optional
        transparency of the metal, default 0.0
    transparency_other : float, optional
        transparency of other atoms, default 0.0
    stick_option : str, optional
        stick options passed to pymol, by default 'set stick_color, grey20'
    render : bool, optional
        run pymol to produce image and serve in jupyter, by default True
    """
    pmpath = pathlib.Path(pmpath)
    pmpath.mkdir(exist_ok=True, parents=True)
    if (pmpath / (render_name + ".pml")).exists():
        render_name = get_next_version_filename(
            render_name, "pml", directory=pmpath.absolute()
        )
        print("File exists, making new file/image with name: ", render_name)
    mol = convert_io_molecule(molecule)
    mol.write_mol2(str(pmpath / (render_name + ".mol2")))
    tag_section = make_tag_section(
        molecule=mol,
        tag_indices=tag_indices,
        tag_color=tag_color,
        tag_radii=tag_radii,
        tag_transparency=tag_transparency,
    )
    render_str = pymol_python_template.format(
        render_name=render_name,
        stick_option=stick_option,
        molecule_buffer=molecule_buffer,
        light_count=light_count,
        metal_scale=metal_scale,
        stick_scale=stick_scale,
        h_scale=h_scale,
        other_scale=other_scale,
        shiny_metal=shiny_metal,
        shiny_other=shiny_other,
        reflect_metal=reflect_metal,
        reflect_other=reflect_other,
        transparency_metal=transparency_metal,
        transparency_other=transparency_other,
        tag_section=tag_section,
        size_x=w,
        size_y=h,
        dpi=dpi,
    )
    with open(pmpath / (render_name + ".pml"), "w") as file1:
        file1.write(render_str)
    if render:
        with change_dir(pmpath):
            pymol_executable = shutil.which("pymol")
            if pymol_executable is not None:
                subprocess.run(
                    [pymol_executable, "-cq", render_name + ".pml"],
                    check=True,
                    capture_output=True,
                )
                display_image_smart(render_name + ".png")
