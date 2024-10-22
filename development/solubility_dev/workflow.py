import pandas as pd
from tqdm import tqdm
import numpy as np
import pathlib
import concurrent.futures
import flux.job
from executorlib import Executor


# encode/decode smiles for filepaths
def encode_smi(insmi):
    outsmi = insmi.replace("/", "x").replace("\\", "z").replace("#", "^")
    return outsmi


def decode_smi(insmi):
    outsmi = insmi.replace("x", "/").replace("z", "\\").replace("^", "#")
    return outsmi


# Up the number of workers
max_workers = 120

# Produced from development workflow ipynb
df = pd.read_csv("filtered_aq_soldb.csv")
smiles = df.SMILES.values

# Get save directory info - make sure matches one in function
savedir = "completed_pickles"
cwd = pathlib.Path(".").absolute()

# Check working dir
print("CWD:", cwd)

# Make the savedirectory
savedir_p = pathlib.Path(savedir)

if not savedir_p.exists():
    savedir_p.mkdir(exist_ok=True, parents=True)

# Check finished in case of restart
done_smiles = [
    decode_smi(x.name.replace(".pkl", "").replace("Smiles_", ""))
    for x in savedir_p.glob("*pkl")
]

# Filter smiles to only not done smiles
to_do = [x for x in smiles if x not in done_smiles]

# Test Smiles - uncomment for easy debugging cases
# to_do = ['C','CC','CCC','CCCC','CCCCC','CCCCO','CON','OCCO']
# to_do += [x+'O' for x in to_do]
# to_do += ['C[O-]','[O-]C']
print("To_DO:", len(to_do))


def full_conf_xtb_workflow(insmiles):
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    from architector.io_calc import CalcExecutor
    from architector import convert_io_molecule
    from architector.io_obabel import generate_obmol_conformers
    from xtb_solvent import xtb_solv_params
    import pandas as pd
    import pathlib

    solvent = "water"
    totaln = 50
    savedir = "completed_pickles"
    savedir_p = pathlib.Path(savedir)

    def gen_confs(smi):
        # Save initial molecule (has charges)
        inmol = convert_io_molecule(smi)
        confs, energies = generate_obmol_conformers(
            inmol, return_energies=True, conf_cutoff=totaln * 100
        )
        # Get the ntotal lowest energy conformers and energies
        energies = np.array(energies)
        confs = np.array(confs)
        inds = np.argsort(energies)[0:totaln]
        energies = energies[inds]
        confs = confs[inds]
        out_confs = []
        # Convert to mol2 strings (mostly to track charge/spin)
        for i, conf in enumerate(confs):
            tmol = convert_io_molecule(conf)
            tmol.charge = inmol.charge
            tmol.uhf = 0
            tmol.xtb_charge = inmol.charge
            tmol.xtb_uhf = 0
            tconf = tmol.write_mol2(
                "UFF_Energy={}".format(energies[i]), writestring=True
            )
            out_confs.append(tconf)
        return out_confs

    # Generate conformers
    confs = gen_confs(insmiles)
    results_list = []
    for conf in confs:
        try:  # Catch errors
            out = CalcExecutor(
                conf,
                method="GFN2-xTB",
                # Store the energy/forces/charges of the minima
                store_results=True,
                relax=True,
                fmax=0.05,  # Tighter force maximum
                xtb_solvent=solvent,
            )
            if out.successful:
                # Store results
                results = out.results
                results["xtb_mol2"] = out.mol.write_mol2(
                    "GFn2-XTB_relax", writestring=True
                )
                results["uff_energy"] = float(
                    conf.split("\n")[1].split("=")[1].split()[0]
                )
                results["uff_mol2"] = conf
                results["smiles"] = insmiles
                results["total_charge"] = out.mol.charge
                results["total_unpaired_electrons"] = out.mol.uhf
                results["n_atoms"] = len(out.mol.ase_atoms)
                results["xtb_solvent"] = solvent
                # THIS call has a subprocess call of XTB
                xtb_sa_eval_dict = xtb_solv_params(results["xtb_mol2"],
                                                   solvent=solvent)
                if isinstance(xtb_sa_eval_dict, dict):
                    results.update(xtb_sa_eval_dict)
                    results["error"] = ""
                    results_list.append(results)
                else:
                    results.update(
                        {
                            "sas": None,
                            "born_radii": None,
                            "error": "XTB Sovlent Eval Error",
                        }
                    )
                    results_list.append(results)
            else:
                results = dict()
                results["energy"] = None
                results["free_energy"] = None
                results["forces"] = None
                results["dipole"] = None
                results["charges"] = None
                results["xtb_mol2"] = None
                results["uff_energy"] = float(
                    conf.split("\n")[1].split("=")[1].split()[0]
                )
                results["uff_mol2"] = conf
                results["smiles"] = insmiles
                results["total_charge"] = None
                results["total_unpaired_electrons"] = None
                results["n_atoms"] = None
                results["xtb_solvent"] = solvent
                results["sas"] = None
                results["born_radii"] = None
                results["error"] = "XTB-Python Relaxation Failed"
                results_list.append(results)
        except:
            results = dict()
            results["energy"] = None
            results["free_energy"] = None
            results["forces"] = None
            results["dipole"] = None
            results["charges"] = None
            results["xtb_mol2"] = None
            results["uff_energy"] = float(
                conf.split("\n")[1].split("=")[1].split()[0])
            results["uff_mol2"] = conf
            results["smiles"] = insmiles
            results["total_charge"] = None
            results["total_unpaired_electrons"] = None
            results["n_atoms"] = None
            results["xtb_solvent"] = solvent
            results["sas"] = None
            results["born_radii"] = None
            results["error"] = "python/XTB error"
            results_list.append(results)

    # encode smiles
    def encode_smi(insmi):
        outsmi = insmi.replace("/", "x").replace("\\", "z").replace("#", "^")
        return outsmi

    # Log output to file from subprocess
    out_df = pd.DataFrame(results_list)
    out_df.to_pickle(savedir_p / ("Smiles_{}.pkl".format(encode_smi(insmiles)))
                     )
    return results_list


with flux.job.FluxExecutor() as flux_exe:
    with Executor(
        max_workers=max_workers,  # total number of cores available to the Executor
        cores_per_worker=1,
        threads_per_core=1,
        cwd=cwd,
        openmpi_oversubscribe=False,  # not available with flux
        flux_executor=flux_exe,
        block_allocation=True,  # reuse existing processes with fixed resources
    ) as exe:
        # Run it
        futs = []
        for td in to_do:
            fut = exe.submit(full_conf_xtb_workflow, td)
            futs.append(fut)
        with tqdm(total=len(to_do)) as pbar:
            for done in concurrent.futures.as_completed(futs):
                lst = done.result()
                pbar.update(1)  # Update pbar
                print("Smiles Done: {}".format(lst[0]["smiles"]))

# Don't need if logging/restart enabled - works well for smaller production runs.
# combined_df = pd.concat([pd.DataFrame(x) for x in out_results])
# combined_df.to_pickle('all_dataframes.pkl')
