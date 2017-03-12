# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:32:02 2016

@author: Brandon
"""

import os
#os.system("python Markowitz_Optimization_Improved.py -d=b -g=0.1")


import subprocess
#subprocess.call("python Markowitz_Optimization_Improved.py -d=b -recov=False -g=0.1", shell=True)

from Markowitz_Optimization_Improved import gamma
print(gamma.value)
