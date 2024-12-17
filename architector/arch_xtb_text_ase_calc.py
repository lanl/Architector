import architector.arch_context_manage as arch_context_manage
import architector.io_molecule as io_molecule

import subprocess as sub
import numpy as np
import shutil

import pathlib
from ase import units
from ase.calculators.calculator import Calculator

methods_dict = {"GFN-FF": "--gfnff", "GFN2-xTB": "-gfn 2", "GFN1-xTB": "--gfn 1"}


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

        super().__init__(
            restart=restart,
            atoms=atoms,
            ignore_bad_restart_file=False,
            label=label,
            **kwargs
        )

        self.parameters = {
            "xtb_method": xtb_method,
            "xtb_solvent": xtb_solvent,
            "xtb_accuracy": xtb_accuracy,
            "xtb_electronic_temperature": xtb_electronic_temperature,
            "xtb_max_iterations": xtb_max_iterations,
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

            if self.parameters.get("xtb_solvent", None) is not None:

                with open("solv_options.txt", "w") as file1:
                    file1.write("$write\n")
                    file1.write("    gbsa=true\n")

                # Run xtb
                execStr = "{} structure.xyz {} --chrg {} --uhf {} --alpb {} -P 1 -a {} --etemp {} --iterations {} --grad -I solv_options.txt> output.xtb".format(
                    xtbPath,
                    methods_dict[self.parameters.get("xtb_method", "GFN2-xTB")],
                    int(charge),
                    int(uhf),
                    self.parameters["xtb_solvent"],
                    self.parameters["xtb_accuracy"],
                    self.parameters["xtb_electronic_temperature"],
                    self.parameters["xtb_max_iterations"],
                )
            else:
                execStr = "{} structure.xyz {} --chrg {} --uhf {} -P 1 -a {} --etemp {} --iterations {} --grad> output.xtb".format(
                    xtbPath,
                    methods_dict[self.parameters.get("xtb_method", "GFN2-xTB")],
                    int(charge),
                    int(uhf),
                    self.parameters["xtb_accuracy"],
                    self.parameters["xtb_electronic_temperature"],
                    self.parameters["xtb_max_iterations"],
                )

            sub.run(
                execStr, shell=True, check=True, stderr=sub.DEVNULL, stdout=sub.DEVNULL
            )

            outpath = pathlib.Path(".")

            self.results = self.read_results(outpath=outpath)

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
        gsolv = None
        for line in lines:
            if ("Gsolv" in line) and ('w/o' not in line):
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

    def read_results(self, outpath):
        """XTB output parser for
        1. energy #
        2. forces #
        3. charges #

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

        chargespath = outpath / "charges"

        charges = None

        if chargespath.exists():
            charges = np.loadtxt(chargespath)
        else:
            chargespath = outpath / "gfnff_charges"
            if chargespath.exists():
                charges = np.loadtxt(chargespath)

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

        if outputpath.exists():
            with open(outputpath, "r") as file1:
                lines = file1.readlines()

            for line in lines:
                if "TOTAL ENERGY" in line:
                    energy = float(line.split()[3]) * units.Ha

        results = self.read_solv_params(lines)

        results.update({"energy": energy, "forces": forces, "charges": charges})

        return results
