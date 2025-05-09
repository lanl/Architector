import architector.arch_context_manage as arch_context_manage
import architector.io_molecule as io_molecule

import subprocess as sub
import numpy as np
import shutil

import pathlib
from ase import units
from ase.calculators.calculator import Calculator

methods_dict = {
    "GFN-FF": "--gfnff",
    "GFN2-xTB": "-gfn 2",
    "GFN1-xTB": "--gfn 1",
}


class XTB_Calculator(Calculator):

    implemented_properties = ["energy", "forces", "charges"]

    def __init__(
        self,
        atoms=None,
        restart=None,
        label="XTB_calulator",
        xtb_method="GFN2-xTB",
        xtb_solvent=None,
        xtb_accuracy=1.0,
        xtb_electronic_temperature=300,
        xtb_max_iterations=250,
        **kwargs
    ):

        super().__init__(restart=restart, atoms=atoms, label=label, **kwargs)

        self.parameters = {
            "xtb_method": xtb_method,
            "xtb_solvent": xtb_solvent,
            "xtb_accuracy": xtb_accuracy,
            "xtb_electronic_temperature": xtb_electronic_temperature,
            "xtb_max_iterations": xtb_max_iterations,
            "opt": kwargs.get("xtb_relax", False),
        }

    def calculate(self, atoms=None, *args, **kwargs):
        """
        Do the calculation

        Parameters
        ----------
        atoms: Atom, optional
            Atom for calculation, if not provided, will use current atom stored
            in the calculator
        """

        if atoms is not None:
            self.atoms = atoms.copy()
        charge = None
        uhf = None
        if hasattr(self.atoms, "info"):
            if isinstance(self.atoms.info, dict):
                charge = self.atoms.info.get("charge", None)
                uhf = self.atoms.info.get("uhf", None)
        if charge is None:  # Read from initial charges
            charge = np.sum(self.atoms.get_initial_charges())
        if uhf is None:  # Read from initial magnetic moments
            uhf = np.sum(self.atoms.get_initial_magnetic_moments())

        # call xtb
        # Convert smiles to xyz string

        xtbPath = shutil.which("xtb")
        xyzstr = io_molecule.convert_ase_xyz(self.atoms)

        with arch_context_manage.make_temp_directory() as _:
            # Write xyz file
            with open("structure.xyz", "w") as outFile:
                outFile.write(xyzstr)

            method = methods_dict[
                self.parameters.get("xtb_method", "GFN2-xTB")
            ].split()

            exec_lst = ["{}".format(xtbPath), "structure.xyz"]
            exec_lst += method

            read_coords = False
            if self.parameters.get("opt", False):
                read_coords = True
                exec_lst += ["--opt"]

            exec_lst += [
                "--chrg",
                "{}".format(int(charge)),
                "--uhf",
                "{}".format(int(uhf)),
                "-P",
                "1",
                "-a",
                "{}".format(self.parameters["xtb_accuracy"]),
                "--etemp",
                "{}".format(self.parameters["xtb_electronic_temperature"]),
                "--iterations",
                "{}".format(self.parameters["xtb_max_iterations"]),
                "--grad",
                "--dipole",
                "--ceasefiles",
            ]

            if self.parameters.get("xtb_solvent", None) is not None:

                with open("solv_options.txt", "w") as file1:
                    file1.write("$write\n")
                    file1.write("    gbsa=true\n")

                exec_lst.append("--alpb")
                exec_lst.append("{}".format(self.parameters["xtb_solvent"]))
                exec_lst.append("-I")
                exec_lst.append("solv_options.txt")

            with open("output.xtb", "w") as file1:

                sub.run(exec_lst, check=True, stderr=sub.DEVNULL, stdout=file1)

            outpath = pathlib.Path(".")

            self.results = self.read_results(
                outpath=outpath, read_coords=read_coords
            )

    def read_solv_params(self, outfilelines):
        """XTB output parser for solvent parameters

        Parameters
        ----------
        outfilelines : str
            file to load

        Returns
        -------
        outdict : dict
        dictionary of output
        """
        lines = outfilelines
        solvent_area_start_key = (
            "generalized Born model for continuum solvation"
        )
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
        gsolv = None
        for line in lines:
            if ("Gsolv" in line) and ("w/o" not in line):
                gsolv = float(line.split()[3]) * units.Ha
                break
        hl_gap = None
        for line in lines:
            if "HOMO-LUMO gap" in line:
                hl_gap = float(line.split()[3])
                break
        outdict = {
            "sas": np.array(sas),
            "born_radii": np.array(born_radii),
            "gsolv_eV": gsolv,
            "hl_gap_eV": hl_gap,
        }
        return outdict

    def read_results(self, outpath, read_coords):
        """XTB output parser for
        1. energy #
        2. forces #
        3. charges #
        4. positions #
        5. covCNs #
        6. dipole #

        on the full relaxation trajectory

        Parameters
        ----------
        outpath : pathlib.Path
            directory to load.

        Returns
        -------
        outdict : dict
        dictionary of output
        """

        forces = None
        gradientspath = outpath / "gradient"

        if gradientspath.exists():
            with open(gradientspath, "r") as file1:
                lines = file1.readlines()
            gradients = []
            for line in lines:
                sline = line.split()
                if len(sline) == 3:
                    gradients.append([float(x) for x in sline])

            gradients = np.array(gradients)

            forces = -1 * gradients * units.Ha / units.Bohr

        outputpath = outpath / "output.xtb"

        energy = None
        coords = None
        charges = None
        covCNs = None
        dipole = None

        if outputpath.exists():
            with open(outputpath, "r") as file1:
                lines = file1.readlines()

            read_charges = False
            read_dipole = False
            for line in lines:
                sline = line.split()
                if "TOTAL ENERGY" in line:
                    energy = float(sline[3]) * units.Ha
                elif len(sline) == 6:
                    if sline[3] == "q":
                        read_charges = True
                        charges = []
                        covCNs = []
                elif read_charges:
                    if len(sline) == 7:
                        charges.append(float(sline[4]))
                        covCNs.append(float(sline[3]))
                    else:
                        read_charges = False
                        charges = np.array(charges)
                        covCNs = np.array(covCNs)
                elif "molecular dipole:" in line:
                    read_dipole = True
                    dipole = []
                elif read_dipole:
                    if "full:" in line:
                        dipole = np.array([float(x) for x in sline[1:4]])
                        read_dipole = False

            if read_coords:
                start = False
                coords = []
                for line in lines:
                    sline = line.split()
                    if "final structure:" in line:
                        start = True
                    elif start and (len(sline) == 0):
                        break
                    elif start and (len(sline) == 4):
                        coords.append([float(x) for x in sline[1:]])
                coords = np.array(coords)

        results = self.read_solv_params(lines)

        results.update(
            {
                "energy": energy,
                "forces": forces,
                "charges": charges,
                "coveCNs": covCNs,
                "dipole": dipole,
            }
        )

        if coords is not None:
            results.update({"positions": coords})

        return results
