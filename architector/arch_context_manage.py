import tempfile
import shutil
import contextlib
import os
import sys

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

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout and stderr output.
    
    This suppresses output at both the Python level (sys.stdout/stderr) and
    the file descriptor level to catch output from C/C++ libraries.
    
    Yields
    ------
    None
    """
    # Save Python-level stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Save file descriptor level stdout/stderr
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    
    devnull = open(os.devnull, 'w')
    
    try:
        # Redirect Python level
        sys.stdout = devnull
        sys.stderr = devnull
        
        # Redirect file descriptor level (catches C/C++ library output)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        
        yield
    finally:
        # Restore file descriptor level first
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        
        # Restore Python level
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()