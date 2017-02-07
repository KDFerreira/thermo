#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:41:59 2017

@author: kevinferreira
"""

from cooling import *
from heating import *
from heatingPoly import *
from dataExplore import *

path = '../data/data.csv'
alpha = 5 #used for time scale. time unit is minute.

print '\n\n\n--------------- cooling ---------------'
neg_RC_inv, RC_var = cooling(path,alpha)
print '\n\nneg_RC_inv = {0}'.format(neg_RC_inv)
print '\nvariance = {0}'.format(RC_var)

print '\n\n\n--------------- heating ---------------'
Qh_C, Qh_C_var = heating(path,alpha,neg_RC_inv)
print '\n\n Qh/C = {0}'.format(Qh_C)
print '\n variance = {0}'.format(Qh_C_var)


print '\n\n\n--------------- Fitting Polynomial ---------------'
Qh_C_start, Qh_C_end  = heatingPolyFit(path,alpha,neg_RC_inv)
print '\n\n Starting Qh/C = {0}'.format(Qh_C_start)
print '\n Ending Qh/C = {0}'.format(Qh_C_end)


print '\n\n\n--------------- Data Deep Dive ---------------'
numNullRows, numT_stp_heat = dataExplore(path)
print '\n\n numNullRows = {0}'.format(numNullRows)
print '\n numT_stp_heat = {0}'.format(numT_stp_heat)
