import json
import re
from ase.db import connect
import numpy as np
import pandas as pd
from tqdm import tqdm

from architector import io_ptable
# import pathlib -> Useful for listing jsons.

def serialize_json_dict(indict):
    """serialize_json_dict ase json serialization routine

    Parameters
    ----------
    indict : dict
        json dictionary

    Returns
    -------
    ss : str
        json serialized version of the 
    """
    ss = '{'
    count = 1
    ids = []
    for key,val in tqdm(indict.items(),total=len(indict)):
        if count % 500 == 0:
            print(count)
        tlines = '"{}"'.format(count) + ': '+ '{\n'
        for skey,sval in val.items():
            if isinstance(sval,str):
                sval = '"{}"'.format(sval)
            tlines += '"{}"'.format(skey)+': '+str(sval)+',\n'
        tlines = tlines.strip().strip(',') + '},\n'
        tlines = re.sub("'",'"',tlines)
        tlines = re.sub('True','true',tlines)
        tlines = re.sub('False','false',tlines)
        if 'nan' in tlines:
            print('nan error')
        else:
            ids.append(count)
            count += 1
            ss += tlines
    ss += '"ids": '+str(ids)+',\n'
    ss += '"nextid": '+str(count)+'}'
    return ss


def merge_JsonFiles(filenamelist,outfname='compiled.json'):
    """jsons = [str(x) for x in p.glob('architector*json')]
    compiled_json_name = 'compiled.json'
    merge_JsonFiles(jsons,compiled_json_name)"""
    result = dict()
    ids = []
    count = 1
    for i,f1 in tqdm(enumerate(filenamelist),total=len(filenamelist)):
        newdata = dict()
        try:
            with open(f1, 'r') as infile:
                data = json.load(infile)
            for key,val in data.items():
                if (key != 'ids') and (key != 'nextid'):
                    newdata[str(count)] = val
                    ids.append(count)
                    count +=1
            result.update(newdata)
        except Exception as e:
            print('Filename: {} Failed'.format(f1))

    with open(outfname, 'w') as output_file:
        ss = serialize_json_dict(result)
        output_file.write(ss)


def convert_arrays_to_npz(dbname, prefix, mindist_cutoff=0.5,
    return_symbols=False, max_force=300, return_df=False):
    """load arrays load ase database into hippynn database arrays
    example : convert_arrays_to_npz('compiled.json','xtbdataset')

    Parameters
    ----------
    filename : str
        filename or path of database to convert
    prefix : str, optional
        prefix for output numpy arrays, by default None
    mindist_cutoff : float, optional
        minimum distance cutoff, default 0.5 Angstroms
    return_symbols : bool, optional
        return the symbols of all structures. Default False
    max_force : float, optional
        cutoff to remove max forces. Default 300 ev/A
    return_df : bool, optional,
        return dataframe with all atoms information, Default False
    """
    try:
        db = connect(dbname)
    except Exception as e:
        print(e)
    record_list = []
    symbols = []
    max_n_atom = 0
    total = None
    with open(dbname, 'r') as f:
        for line in f:
            pass
        last_line = line
    print(last_line)
    total = int(last_line.split()[1].split('}')[0])
    any_pbc = False
    for row in tqdm(db.select(),total=total):
        is_pbc = False
        if np.any(row.pbc):
            is_pbc = True
            any_pbc = True
        try:
            syms,counts=np.unique(row.symbols,return_counts=True)
            result_dict = {
                'atoms':row.numbers,
                'xyz': row.positions,
                'cell': row.cell,
                'is_pbc':is_pbc,
                'force': row.forces,
                'energy': row.energy,
                'atomization_energy':row.energy -  np.sum([io_ptable.xtb_single_atom_ref_es[sym]*counts[i] for i,sym in enumerate(syms)]),
                'uid': row.unique_id,
                'relaxed': row.relaxed,
                'geo_step': row.geo_step
            }
            distmat = row.toatoms().get_all_distances() + np.eye(len(row.numbers))* mindist_cutoff*2
            maxforce = np.max(np.linalg.norm(row.forces,axis=1))
            if (distmat.min() > mindist_cutoff) and (maxforce < max_force): # Hard distance cutoff, forces cutoff
                record_list.append(result_dict)
                if row.natoms > max_n_atom:
                    max_n_atom = row.natoms
                if return_symbols:
                    symbols += row.symbols
        except:
            pass
    n_record = len(record_list)
    # Sort the list base on number of atoms
    record_list.sort(key=lambda rec: len(rec['atoms']))
    # Save the record uid from the ase db object names.
    xyz_array = np.zeros([n_record, max_n_atom, 3])
    force_array = np.zeros([n_record, max_n_atom, 3])
    atom_z_array = np.zeros([n_record, max_n_atom])
    if any_pbc:
        cell_array = np.zeros([n_record, 3, 3])
        pbc = True
    else:
        cell_array = np.zeros([n_record, 3, 3])
        pbc = False
    energy_array = np.array([record['energy'] for record in record_list])
    atomization_array = np.array([record['atomization_energy'] for record in record_list])
    for i, record in enumerate(record_list):
        natom = len(record['atoms'])
        xyz_array[i, :natom, :] = record['xyz']
        force_array[i, :natom, :] = record['force']
        atom_z_array[i,:natom] = record['atoms']
        if pbc:
            cell_array[i, :, :] = record['cell']
    np.save("data-" + prefix + 'atomization_energy.npy', atomization_array) 
    np.save("data-" + prefix + 'energy.npy', energy_array)        
    np.save("data-" + prefix + 'R.npy', xyz_array)
    np.save("data-" + prefix + 'force.npy', force_array)
    np.save("data-" + prefix + 'Z.npy', atom_z_array.astype('int'))
    if return_symbols:
        return symbols
    if return_df:
        return pd.DataFrame(record_list)