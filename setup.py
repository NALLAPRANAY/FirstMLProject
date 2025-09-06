from setuptools import find_packages,setup
from typing import List
Hypen_e_dot="-e."

def get_requirments(file_path:str)->List[str]:
    '''
    This function will return the list of requirments
    '''
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req.replace("\n","") for req in requirments]
        if Hypen_e_dot in requirments:
            requirments.remove(Hypen_e_dot)
    return requirments 

setup(
    name='firstmlproject',
    version='0.0.1',
    author='PRANAY NALLA',
    author_email='pranay6243@gmail.com',
    packages=find_packages(),
    install_requires=get_requirments('requirments.txt')
)