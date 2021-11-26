import os
import sys
import yaml
import random
import shutil
import subprocess
from typing import List, Tuple
from contextlib import contextmanager
from time import time
from datetime import datetime

import ipdb
import numpy as np
import torch


def set_seed(seed: int) -> None:
    r"""
    Set the seed of the system.

    Args: 
        seed (int): The artificial random seed.
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@contextmanager
def nullcontext():
    yield


def set_trace() -> None:
    ipdb.set_trace()


def yaml_load(file: str, mode: str = "r", Loader = yaml.FullLoader, **kwargs):
    r"""
    Parse the YAML document and produce the corresponding Python object.

    Args:
        file (str): Path of the YAML document.
        mode (str): Mode to open the document.
        Loader: Loader for YAML.
        kwargs: Other options for opening the document.
    """
    with open(file, mode, **kwargs) as rf:
        data = yaml.load(rf, Loader = yaml.FullLoader)
    return data


def yaml_dump(data, file: str, mode: str = "w", **kwargs) -> None:
    r"""
    Serialize a Python object into a YAML document.

    Args:
        data: The Python object to be serialized.
        file (str): Path of the YAML document.
        mode (str): Mode to open the document.
        kwargs: Other options for opening the document.
    """
    with open(file, mode, **kwargs) as wf:
        yaml.dump(data, wf)


def remove_files(root: str, target: str) -> int:
    r"""
    Delete the target files under the given root recursively.

    Args:
        root (str): Root.
        target (str): Name of the target files.

    Returns:
        delete_num (int): Number of the deleted files.

    Examples::
        >>> # Remove files named "__init__.py" under the current path recursively
        >>> delete_num = remove_files("./", "__init__.py")
    """
    delete_num = 0
    for _root, dirs, files in os.walk(root):
        for name in files:
            if name == target:
                delete_num += 1
                os.remove(os.path.join(_root, name))
    return delete_num


def remove_dirs(root: str, target: str) -> int:
    r"""
    Delete the target dirs under the given root recursively.

    Args:
        root (str): Root.
        target (str): Name of the target dirs.

    Returns:
        delete_num (int): Number of the deleted dirs.

    Examples::
        >>> # Remove all dirs named "__pycache__" under the current path recursively
        >>> delete_num = remove_dirs("./", "__pycache__")
    """
    delete_num = 0
    for _root, dirs, files in os.walk(root, topdown = False):
        for name in dirs:
            if name == target:
                delete_num += 1
                shutil.rmtree(os.path.join(_root, name))
    return delete_num


def remove_targets(root: str, target: str) -> int:
    r"""
    Delete the target files and dirs under the given root recursively.

    Args:
        root (str): Root.
        target (str): Name of the target files or dirs.

    Returns:
        delete_num (int): Number of the deleted files or dirs.

    Examples::
        >>> # Remove dirs or files named "__pycache__" under the current path recursively
        >>> delete_num = remove_targets("./", "__pycache__")
    """
    delete_num = remove_files(root, target) + remove_dirs(root, target)
    return delete_num
    

class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, a_callable):
        a_callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(a_callable)


def run_processes(commands: List[str], **kwargs) -> List[str]:
    r"""
    Run a list of commands one-by-one.

    Args:
        command (List[str]): A list of commands to be run.
        kwargs: Configurations except 'shell' for `subprocess.check_call`.
                We note that `shell` must be `True`  for this function.

    Returns:
        unsuccessful_commands (List[str]): A list of commands that fail to run successfully.

    Examples::
        >>> # Assumes a runnable python file named `test.py` exists
        >>> # And assumes no file named `not_exist.py` exists
        >>> # Now, we want to run `test.py` and `not_exist.py` one-by-one
        >>> commands = ["python test.py", "python not_exist.py"]
        >>> unsuccessful_commands = run_processes(commands) # ["python not_exist.py"]
        >>> print("Unsuccessful commands: {}".format(unsuccessful_commands))
    """
    unsuccessful_commands = []
    for command in commands:
        try:
            subprocess.check_call(command, shell = True, **kwargs)
        except subprocess.CalledProcessError:
            unsuccessful_commands.append(command)
    return unsuccessful_commands


def makedir(path: str, remove: bool = False, quiet: bool = False) -> None:
    r"""
    Create a leaf directory and all intermediate ones.

    Args:
        path (str): The directory to be made.
        remove (bool): Sometimes, the targeted directory has existed. If `remove` is false, 
                       the targeted directory will not be removed. Default: False.
        quiet (bool): When the targeted directory has existed and `remove == True`:
                      If `quiet == False`, we will ask whether to remove the existing directory.
                      Else, the existing directory will be removed directly. Default: False.
    """
    if os.path.exists(path) and remove:
        if not quiet:
            response = input(
                "The {} already exists.".format(path) + \
                "Do you want to delete it anyway. [Y/y/yes] or [N/n/no/others], default is N\n"
            )
            if str(response) not in ["Y", "y", "yes"]:
                print("exit!")
                sys.exit(0)    
        shutil.rmtree(path)
    
    if not os.path.isdir(path):
        os.makedirs(path)


class TimeEstimator(object):
    r"""
    Remaining time estimator.
    Estimate the remaining time to finish all iterations based on history.

    Args:
        iter_num (int): Total iteration number.
    """
    def __init__(self, iter_num: int) -> None:
        self.iter_num = iter_num
        self.start_time = datetime.now()
        self.last_time = datetime.now()
        self.current_iter = 0

    def step(self) -> Tuple[datetime, datetime]:
        r"""
        Update the estimator and return the estimated remaining time.

        Returns:
            overall_time (datetime): The remaining time estimated based on the whole history.
            nearest_time (datetime): The remaining time estimated based on the last iteration.
        """
        assert self.current_iter <= self.iter_num, "The maximum number of iterations is exceeded."
        self.current_iter += 1
        current_time = datetime.now()
        overall_time = (current_time - self.start_time) / self.current_iter * (self.iter_num - self.current_iter)
        nearest_time = (current_time - self.last_time) * (self.iter_num - self.current_iter)
        self.last_time = current_time
        return overall_time, nearest_time
