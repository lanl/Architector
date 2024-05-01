# Execute with:
# mpiexec -n 12 python -m mpi4py.futures mpirun.py
# For example

import os

# Note - these are needed at this level as well to force the correct number of threads to xTB.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import as_completed
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import time
import pathlib
import pickle
import subprocess

outpath = 'outputs'
input_path = '../meta_production_sample.pkl'


def calc(input_dict):
    with open(input_dict['name']+'_running','w') as file1:
        file1.write(str(datetime.now()))
    pickle_input = pickle.dumps(input_dict,
                                0).decode(encoding='latin-1')
    start = time.time()
    try:
        subprocess.check_output(["python", "run_script.py", pickle_input],
                                universal_newlines=True)
    except subprocess.CalledProcessError as e:
        end = time.time()
        with open(os.path.join(outpath,
                               inp_dict['name']+'_failed.txt'),
                               'w') as file1:
            file1.write('Total_Time: {}\n'.format(end-start))
            file1.write('Failed at : {}\n'.format(datetime.now()))
            file1.write('Subprocess error: {}\n'.format(e))
        pass
    os.remove(input_dict['name']+'_running')
    return True


if __name__ == '__main__':
    with MPIPoolExecutor() as exe:
        if exe is not None:

            if not os.path.exists(outpath):
                os.mkdir(outpath)
            # Load input dataframe
            indf = pd.read_pickle(input_path)

            # Add index as name of job from input dataframe.
            newindf_rows = []
            for i, row in indf.iterrows():
                inp_dict = row['architector_input']
                inp_dict['name'] = str(i)
                newindf_rows.append(inp_dict)

            # Check the output path to not duplicate 
            # Finished/failed architector runs.
            op = pathlib.Path(outpath)
            done_list = [p.name.replace('.pkl',
                                    '').replace('_failed.txt',
                                                '') for p in op.glob('*')]
            futs = []
            for d in newindf_rows:
                if d['name'] not in done_list:
                    # print('Submitting: {}'.format(i))
                    fut = exe.submit(calc, d)
                    futs.append(fut)
            # Track progress of completed jobs.
            for x in tqdm(as_completed(futs), total=len(futs)):
                res = x.result()
