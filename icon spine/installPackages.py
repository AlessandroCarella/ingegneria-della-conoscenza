import subprocess
import sys
from tqdm import tqdm

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def installPackages ():
    packages = []
    with open ("requirements.txt", "r") as file:
        packages = file.readlines()
    
    try:
        print ("Installing needed packages")
        for package in tqdm(packages):
            install(package)
    except:
        try:
            print ("Somethingh went wrong, checking installed packages")
            for package in tqdm(packages):
                install(package)
        except:
            print ("Not able to install required packages, try via pip command in terminal, the required packages are in the requirements.txt file")
