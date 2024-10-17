import architector.arch_context_manage as arch_context_manage
import architector.io_molecule as io_molecule
import shutil
import subprocess as sub
import numpy as np


def read_solv_params(file):
    with open(file, "r") as file1:
        lines = file1.readlines()
    solvent_area_start_key = "generalized Born model for continuum solvation"
    sovlent_area_end_key = "total SASA"
    start = False
    sas = []
    born_radii = []
    for line in lines:
        if solvent_area_start_key in line:
            start = True
        elif sovlent_area_end_key in line:
            break
        elif start and (len(line.strip().split()) == 6):
            sline = line.strip().split()
            if sline[1].isnumeric():
                sas.append(float(sline[4]))
                born_radii.append(float(sline[3]))
        else:
            pass
    outdict = {"sas": np.array(sas), "born_radii": np.array(born_radii)}
    return outdict


def xtb_solv_params(structure, solvent="water"):
    """
    Take in a structure, evaluate with xtb to SA surface area

    Parameters
    ----------
    structure : mol2str
        structure passsed
    solvent : str, optional
        whether to use a solvent for conformer evalulation, default 'none'

    Returns
    ----------
    results : dict
        Compiled XTB results
    """

    # Convert smiles to xyz string

    xtbPath = shutil.which("xtb")

    mol = io_molecule.convert_io_molecule(structure)
    if mol.xtb_charge is None:
        mol.detect_charge_spin()
    mol_charge = mol.xtb_charge

    mol.swap_actinide()

    even_odd_electrons = (
        np.sum([atom.number for atom in mol.ase_atoms]) - mol_charge
    ) % 2
    if mol.xtb_uhf is not None:
        uhf = mol.xtb_uhf
    else:
        uhf = 0  # Set spin to LS by default
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

    mol_charge = int(mol_charge)  # Ensure integers
    uhf = int(uhf)
    xyzstr = io_molecule.convert_ase_xyz(mol.ase_atoms)

    with arch_context_manage.make_temp_directory() as _:
        # Write xyz file
        with open("structure.xyz", "w") as outFile:
            outFile.write(xyzstr)

        with open("solv_options.txt", "w") as file1:
            file1.write("$write\n")
            file1.write("    gbsa=true\n")

        # Run xtb
        execStr = "{} structure.xyz --chrg {} --uhf {} --alpb {} -P 1 -I solv_options.txt> output.xtb".format(
            xtbPath, int(mol_charge), int(uhf), solvent
        )

        sub.run(execStr, shell=True, check=True)

        # Read conformers from file
        result_dict = read_solv_params("output.xtb")

    return result_dict
