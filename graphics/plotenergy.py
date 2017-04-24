import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
import matplotlib.pyplot as plt

interpolate = False
interpolate = True
usetrueminimas = True
usetrueminimas = False
try:
    script, mep_values, fName = argv
except:
    print("ERROR: You didn't provide the params file for the programm", argv[0])

energy_path = np.loadtxt(mep_values)
minimums = argrelextrema(energy_path, np.less_equal)[0]
print(energy_path[minimums])
maxima = argrelextrema(energy_path, np.greater)[0]
print(energy_path[maxima])

#if(usetrueminimas):
#    energy_path=energy_path[minimums[0]:minimums[-1]]
#
fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax.set_xticklabels([]) # disable ticks 
#ax.set_aspect('auto')
M = energy_path.shape[0]
#plt.yticks(np.arange(min(energy_path), max(energy_path)+0.4, 0.1))
plt.ylim([np.min(energy_path) - .06, np.max(energy_path) + .4])
plt.ylabel('eV')
plt.xlim([0-5, M+7])
x = range(0, M)
y = energy_path
plt.plot(x, y, linewidth=3)
#f2 = interp1d(x, y, kind='cubic')
#xnew = np.linspace(0, M-1, num=1000, endpoint=True)
#
#if(interpolate):
#    plt.plot(xnew, f2(xnew), linewidth=3)
#else:
#    plt.plot(x, y, linewidth=3)
#
#rn = ['I','II','III','IV','V','VI','VII']
#
#if(usetrueminimas):
#    minimums = (minimums-minimums[0]) #shift the index of minimas to match new size of enrgy
#    minimums[-1] = minimums[-1]-1
#
#ax.plot([minimums], [energy_path[minimums]], 'o', color='green')
#for i in range(len(minimums)):
    #ax.annotate(rn[i], xy=(minimums[i], energy_path[minimums[i]]), xytext=(minimums[i]-2, energy_path[minimums[i]]-0.04))#, arrowprops=dict(facecolor='black', shrink=0.05),)
    #ax.annotate(rn[i], xy=(minimums[i], energy_path[minimums[i]]), xytext=(minimums[i]-0.2, energy_path[minimums[i]]-0.006))#, arrowprops=dict(facecolor='black', shrink=0.05),)
plt.savefig(("mep-"+fName+".eps"),bbox_inches='tight', dpi=300)
