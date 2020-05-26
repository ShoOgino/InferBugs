import sys
import os
import csv
import glob
import pandas as pd
import re

def checkRepositoryHasCorrectFiles():
    pathFolderRepository=sys.argv[1]

    listPaths=[p for p in glob.glob(os.path.abspath(pathFolderRepository) + "/**/*.mjava", recursive=True) if re.search('^(?!.*(unit|tasks)).*$', p)]
    mergePatterns(Project)
    print(listPaths)
    print(len(listPaths))

checkRepositoryHasCorrectFiles()