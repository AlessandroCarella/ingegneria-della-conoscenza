import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def installPackages ():
    packages = []
    with open ("requirements.txt", "r") as file:
        packages = file.readlines()
    for package in packages:
        install(package)