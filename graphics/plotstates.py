# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi
from numpy import cos
from numpy import sin
from numpy import sqrt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.signal import argrelextrema

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json
from sys import argv
import geometry as g

output_path = ""#"data/"
output_path_img = ""#"img/"

try:
    script, vcoords_file, outputname = argv
except:
    print("ERROR: You didn't provide the all params for the programm", argv[0], "it should be params vectorcoords")



vectorcoords = g.load_vectorcoordinates_from_txt(vcoords_file)
centerscoords = g.generate_centercoords(vectorcoords)
angle_against_ox = np.rad2deg(g.generate_angle_with_Ox(vectorcoords))

energy_path = np.loadtxt("energy.txt")
path = np.loadtxt("path.txt")
#path = np.loadtxt("pp")
#energy_path = np.loadtxt("ep")

minm = argrelextrema(energy_path,np.less_equal)
#print('min',energy_path[minm])
#minm = argrelextrema(energy_path,np.greater)
#print('max',energy_path[minm])
pt = ((path[:,minm])[:,0]).T
pt = np.loadtxt("pp").T

m = len(pt)
gs1 = gridspec.GridSpec(1,m)
gs1.update(wspace=0.025, hspace=0.05)

def add_system_state(state, fig, ccoords):
    n = len(state)
    r = 1 # radius of the hexagon ring
    il = r*1.
    iw = il*0.3
    ibw = il*0.7
    arrow_width = 0.025
    arrow_length = 1.

    plt.xlim(-r-.3, r+.3)
    plt.ylim(-r-.3, r+.3)

    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xticklabels([]); ax.set_yticklabels([]) # disable ticks 
    plt.axis('off') # disable axes


    # Define positions for centers of hexagon edges
    XX = list(zip(*ccoords))[0]
    YY = list(zip(*ccoords))[1]
    UU = [0.]*n
    VV = [arrow_length]*n
    plt.quiver(XX,YY,UU,VV,angles=((aa)*180/pi), pivot="middle", scale=4.0, width=.025)

    #_t_angle=-120
    for i in range(n):
        if(i ==8): il=il*2 #TODO:replace with automatic things
        #_t_angle += 120
        _t_angle_n = angle_against_ox[i]
        cp = ccoords[i]
        shape = patches.Ellipse(cp, il, iw, angle = 0, color = '.75',zorder=-1)
        cdb = shape.get_extents()
        t_start = ax.transData
        #t = mpl.transforms.Affine2D().rotate_deg_around(cp[0], cp[1], _t_angle)
        t = mpl.transforms.Affine2D().rotate_deg_around(cp[0], cp[1], _t_angle_n)
        t_end = t+t_start
        island = patches.FancyBboxPatch((cdb.xmin, cdb.ymin), abs(cdb.width), abs(cdb.height),transform = t_end,  boxstyle = patches.BoxStyle.Round(pad=0.00, rounding_size=.15),  fill = None, ls='solid', color = '0', lw=ibw)
        plt.gca().add_patch(island)

#----------- Islands on a hexagon grid ----------
#ax = plt.subplot2grid(grid_size, (1,0), rowspan=1, colspan=2)
rn = ['I','II','III','IV','V','VI','VII','VIII','IX', 'X', 'XI', 'XII']
for i in range(m):
    ax = plt.subplot(gs1[i])
    aa = pt[i]
    add_system_state(aa,ax, centerscoords)
    #ax.text(0.5, 0.5, rn[i], horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

plt.savefig(output_path_img+outputname+".pdf",bbox_inches='tight', dpi=300)

#
##plt.subplot(222)
#plt.subplot2grid(grid_size, (0,4), rowspan=1, colspan=4)
#plt.ylim([np.min(energy_path) - .1, np.max(energy_path) + .1])
#plt.ylabel('eV')
#plt.xlim([0-3, path.shape[1]+2])
#plt.plot(range(0, path.shape[1]), energy_path, linewidth=2)
#plt.tight_layout()
