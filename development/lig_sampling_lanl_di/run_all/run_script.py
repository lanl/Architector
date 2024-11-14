import os
import pandas as pd
import time
# import datetime
from architector import build_complex
import sys
import pickle

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

outpath = 'outputs'
pickle_input = sys.argv[1]
inp_dict = pickle.loads(pickle_input.encode("latin-1"))

def run_static(inp_dict):
    # Keeping track of running for debugging.
    # with open(inp_dict['name']+'_started', 'w') as file1:
    #     file1.write('started at: {}'.format(datetime.datetime.now()))
    start = time.time()
    complex_mol = []
    try:
        complex_mol = build_complex(inp_dict) # Run architector
        end = time.time()
    except:  # Catch all manner of errors.
        end = time.time()
        with open(os.path.join(outpath,
                               inp_dict['name']+'_failed.txt'),
                               'w') as file1:
            file1.write('Total_Time: {}\n'.format(end-start))
        return inp_dict['name'], False
    else:
        ttime = end-start
        dfrows = []
        for key, val in complex_mol.items():
            val['total_walltime'] = ttime
            val['gen_unique_name'] = key
            dfrows.append(val)
        resultsdf = pd.DataFrame(dfrows)
        # Don't save the ase_atoms.
        if "ase_atoms" in resultsdf.columns:
            resultsdf.drop("ase_atoms", inplace=True, axis=1)
        resultsdf.to_pickle(os.path.join(outpath,
                                         inp_dict['name']+'.pkl'))
        return inp_dict['name'], True


out = run_static(inp_dict)
