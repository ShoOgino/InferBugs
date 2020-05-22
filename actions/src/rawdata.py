import glob
import os
import sys
import subprocess

class Rawdata:
    def __init__(self,urlRepository, pathData, nameRepository, filterFile, codeIssueJira, nameProjectJira):
        pass
    def prepare(self):
        cloneRepository(self, urlRepository, pathRepository)
        checkoutRepository(self, pathRepository, identifierDataset.IdCommit)
    def local_chdir(self, func):
        def _inner(self, *args, **kwargs):
            dir_original = os.getcwd()
            ret = func(*args, **kwargs)
            os.chdir(dir_original)
            return ret
        return _inner
    def cloneRepository(self, urlRepository, pathRepository):
        cmdGitClone=["git", "clone"]
        cmdGitClone.extend([urlRepository, pathRepository])
        subprocess.call()
    @local_chdir
    def checkoutRepository(self, pathRepository, IdCommit):
        os.chdir(pathRepository)
        cmdGitCheckout=["git", "checkout"]
        cmdGitCheckout.extend([IdCommit])
    def save(self):
        pass
    def load(self):
        pass
