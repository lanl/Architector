import tempfile
import shutil
import contextlib
import os

@contextlib.contextmanager
def make_temp_directory(prefix=None):
    """make_temp_directory function to make a temporary directory and change there.
    Very useful for running on supercomputers with scratch!

    Parameters
    ----------
    prefix : str, optional
        path prefix to temporary folder, by default None

    Yields
    ------
    temp_dir : str
        name of the temporary directory
    """
    mycwd = os.getcwd()
    try:
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(mycwd)
        shutil.rmtree(temp_dir)
        