# Entry script to run an experiment.
# Just a simple script to handle the correct 'project' directory
# (if the packages were not build with pip or poetry, or something else).

# This part of the script adapts the Python sys.path so the karting_challenge_experiment package
# can be used like a package without packaging it.
# This is meant for development use-cases only.
# When development and debugging is done, the packages can be build with and installed via pip
# or conda and this part can be removed.
import sys
import os

# Get the absolute path to the 'project' directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'project' directory to the Python path
sys.path.append(project_dir)

# This part of the script is the main part that is calling the main function
# of the karting_challenge_experiment package.
from karting_challenge_experiment.main import main

if __name__ == '__main__':
    main()
