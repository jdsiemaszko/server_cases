import numpy as np 
import timeit
import matplotlib.pyplot as plt
from pathlib import Path
import pHyFlow
from solvers.dipole_initial import Dipole

import os
import sys
import yaml
import re
import csv
import pandas


#---------------Current directory and paths---------------------F---------------
arg = sys.argv
if len(arg) > 2:
    raise Exception("More than two arguments inserted!")
if len(arg) <= 1:
    raise Exception("No config file specificed!")
configFile = arg[1]

#-----------------------Config the yaml file ------------------
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
config = yaml.load(open(os.path.join(configFile)),Loader=loader)

start_index = configFile.index('_') + 1
end_index = configFile.index('.')
# case = configFile[start_index:end_index]
case = config['case']

case_dir = os.path.join(os.getcwd(), 'results', case)
data_dir = os.path.join(case_dir,config["data_folder"])
plots_dir = os.path.join(case_dir,config["plots_folder"])

Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(plots_dir).mkdir(parents=True, exist_ok=True)

# #--------------Copy values from the config file--------------------

nPlotPoints = config["nPlotPoints"]
xMinPlot = config["xMinPlot"]
xMaxPlot = config["xMaxPlot"]
yMinPlot = config["yMinPlot"]
yMaxPlot = config["yMaxPlot"]

writeInterval_plots = config["writeInterval_plots"]
plot_flag = config["plot_flag"]

vInfx = config["vInfx"]
vInfy = config["vInfy"]
vInf = np.array([vInfx, vInfy])

nu = config["nu"]

x1 = config['x1']
y1 = config['y1']
x2 = config['x2']
y2 = config['y2']
R = config['R']

Gamma = config['Gamma']

xMin = config['xMin']
xMax = config['xMax']
yMin = config['yMin']
yMax = config['yMax']

coreSize = config["coreSize"]

overlap = config["overlap"]

deltaTc = config["deltaTc"]
nTimeSteps = config["nTimeSteps"]

compressionFlag = config['compressionFlag']
compression_method = config["compression_method"]
support_method = config["support_method"]
compression_params = config["compression_params"]
support_params = config["support_params"]

compression_stride = config['compression_stride']

#--------------------Plot parameters--------------------------------
xplot,yplot = np.meshgrid(np.linspace(xMinPlot,xMaxPlot,nPlotPoints),np.linspace(yMinPlot,yMaxPlot,nPlotPoints))
xplotflat = xplot.flatten()
yplotflat = yplot.flatten()
xyPlot = np.column_stack((xplotflat, yplotflat))

#------------------Parameters for blobs-----------------------------
computationParams = {'hardware':config['hardware'], 'method':config['method']}

blobControlParams = {'methodPopulationControl':config['method_popControl'],'typeOfThresholds':'relative', 'stepRedistribution':config['stepRedistribution'],\
                     'stepPopulationControl':config['stepPopulationControl'], 'gThresholdLocal': config['gThresholdLocal'],\
                     'gThresholdGlobal':config['gThresholdGlobal']}

blobDiffusionParams = {'method' : config['method_diffusion']}

timeIntegrationParams = {'method':config['time_integration_method']}

kernelParams = {'kernel' : config['kernel'], 'coreSize' : config['coreSize']}

avrmParams = config['avrm_params']

compressionParams = {'method':compression_method, 'support':support_method, 'methodParams':compression_params,\
                     'supportParams':support_params}

xShift = config['xShift'] 
yShift = config['yShift']
#-------------------------------------------------------------------

hv = np.sqrt(deltaTc * nu)
sigmaInit = float(overlap * np.sqrt(12) * hv)
hBlob = sigmaInit / overlap

initial_dipole = Dipole(Gamma, nu, x1, y1, x2, y2, R, vInfx, vInfy)
del Dipole
xBlob, yBlob = np.meshgrid(np.arange(xMin, xMax, hBlob), np.arange(yMin, yMax, hBlob))
xBlobFlat = xBlob.flatten()
yBlobFlat = yBlob.flatten()
# xyBlobs = np.column_stack((xBlobFlat,yBlobFlat))

g = initial_dipole.vorticity(xBlobFlat, yBlobFlat) * hBlob * hBlob

wField = (xBlobFlat, yBlobFlat, g)

if coreSize == 'fixed':
    sigma = sigmaInit
else:
    sigma = np.full(xBlobFlat.shape, sigmaInit)

blobs = pHyFlow.blobs.Blobs(wField,vInf,nu,deltaTc,sigma,overlap,xShift,yShift,
                            kernelParams=kernelParams,
                            diffusionParams=blobDiffusionParams,
                            velocityComputationParams=computationParams,
                            timeIntegrationParams=timeIntegrationParams,
                            blobControlParams=blobControlParams,
                            avrmParams=avrmParams,
                            mergingParams=compressionParams)


header = ['Time', 'NoBlobs', 'Evolution_time', 'Circulation']
with open('{}/times_{}.csv'.format(data_dir,case), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

ux, uy = blobs.evaluateVelocity(xplotflat,yplotflat)
omega = blobs.evaluateVorticity(xplotflat,yplotflat)
np.savetxt(os.path.join(data_dir,"results_{}_{n:06d}.csv".format(case,n=0)), np.c_[xplotflat,yplotflat,ux,uy,omega], delimiter=' ')
if coreSize == 'variable':
    np.savetxt(os.path.join(data_dir, "blobs_{}_{n:06d}.csv".format(case,n=0)), np.c_[blobs.x, blobs.y, blobs.g, blobs.sigma], delimiter= ' ')
else :
    np.savetxt(os.path.join(data_dir, "blobs_{}_{n:06d}.csv".format(case,n=0)), np.c_[blobs.x, blobs.y, blobs.g], delimiter= ' ')
blobs.populationControl()
for timeStep in range(1, nTimeSteps+1):
    time_start = timeit.default_timer()
    blobs.evolve()
    if (compressionFlag and timeStep%compression_stride == 0):
        print('----------------Performing Compression--------------')
        nbefore = blobs.numBlobs
        blobs._compress()
        nafter = blobs.numBlobs
        print(f'removed {nbefore-nafter} particles')
        print(f'current number of particles: {nafter}')
    time_end = timeit.default_timer()
    print("Time to evolve in timeStep {} is {}".format(timeStep,time_end - time_start))

    evolution_time = time_end - time_start
    T = timeStep*deltaTc

    data = [T,blobs.numBlobs,evolution_time, blobs.g.sum()]

    with open('{}/times_{}.csv'.format(data_dir,case), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    if timeStep%writeInterval_plots==0 :
        ux, uy = blobs.evaluateVelocity(xplotflat,yplotflat)
        omega = blobs.evaluateVorticity(xplotflat,yplotflat)
        print(np.max(np.abs(omega)))

        np.savetxt(os.path.join(data_dir,"results_{}_{n:06d}.csv".format(case,n=timeStep)), np.c_[xplotflat,yplotflat,ux,uy,omega], delimiter=' ')
        if coreSize == 'variable':
            np.savetxt(os.path.join(data_dir, "blobs_{}_{n:06d}.csv".format(case,n=timeStep)), np.c_[blobs.x, blobs.y, blobs.g, blobs.sigma], delimiter= ' ')
        else :
            np.savetxt(os.path.join(data_dir, "blobs_{}_{n:06d}.csv".format(case,n=timeStep)), np.c_[blobs.x, blobs.y, blobs.g], delimiter= ' ')

if plot_flag == True:
    uxNorm = np.array([])
    uyNorm = np.array([])
    omegaNorm = np.array([])
    t_norm = np.array([])
    #Line plots
    times_file = os.path.join(data_dir,"times_{}.csv".format(case))
    times_data = pandas.read_csv(times_file)

    time = times_data['Time']
    noBlobs = times_data['NoBlobs']
    evolution_time = times_data['Evolution_time']
    circulation = times_data['Circulation']

    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(time,noBlobs, label='No of Particles')
    plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title('Total number of particles')
    plt.ylabel('Particles')
    plt.xlabel('time $(sec)$')
    plt.legend()
    plt.savefig("{}/number_of_particles_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(time,circulation, label='Circulation deficit')
    plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title('absolute error in circulation')
    plt.ylabel('circulation')
    plt.xlabel('time $(sec)$')
    plt.legend()
    plt.savefig("{}/circulation_error_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

    fig = plt.subplots(figsize=(6,6))
    index = np.arange(len(evolution_time))
    width = deltaTc
    lagrangian = plt.bar(index[1:]*deltaTc, evolution_time[1:], width)
    plt.ylabel('Time (s)')
    plt.xlabel('Simulation time (s)')
    plt.title('Evolution time')
    plt.savefig("{}/times_{}.png".format(plots_dir,case), dpi=300, bbox_inches="tight")

    for timeStep in range(0, nTimeSteps+1):
        if timeStep%writeInterval_plots == 0:
            ####Fields
            lagrangian_file = os.path.join(data_dir,'results_{}_{n:06d}.csv'.format(case,n=timeStep))
            lagrangian_data = np.genfromtxt(lagrangian_file)

            xplot = lagrangian_data[:,0]
            yplot = lagrangian_data[:,1]
            length = int(np.sqrt(len(xplot)))
            xPlotMesh = xplot.reshape(length,length)
            yPlotMesh = yplot.reshape(length,length)

            lagrangian_ux = lagrangian_data[:,2]
            lagrangian_uy = lagrangian_data[:,3]
            lagrangian_omega = lagrangian_data[:,4]

            xTicks = np.linspace(-2,2,5)
            yTicks = np.linspace(-2,2,5)

            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.set_aspect("equal")
            ax.set_xticks(xTicks)
            ax.set_yticks(yTicks)
            plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cax = ax.contourf(xPlotMesh,yPlotMesh,lagrangian_omega.reshape(length,length),levels=100,cmap='RdBu',extend="both")
            cbar = fig.colorbar(cax,format="%.4f")
            cbar.set_label("Vorticity (1/s)")
            plt.tight_layout()
            plt.savefig("{}/vorticity_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.set_aspect("equal")
            ax.set_xticks(xTicks)
            ax.set_yticks(yTicks)
            plt.grid(color = '#666666', which='major', linestyle = '--', linewidth = 0.5)
            plt.minorticks_on()
            plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cax = ax.contourf(xPlotMesh,yPlotMesh,lagrangian_ux.reshape(length,length),levels=100,cmap='RdBu',extend="both")
            cbar = fig.colorbar(cax,format="%.4f")
            cbar.set_label("Velocity (1/s)")
            plt.tight_layout()
            plt.savefig("{}/velocity_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
            plt.close(fig)


#### Blobs distribution

            blobs_file = os.path.join(data_dir,'blobs_{}_{n:06d}.csv'.format(case,n=timeStep))
            blobs_data = np.genfromtxt(blobs_file)

            if len(blobs_data.shape) == 1:
                blobs_x = blobs_data[0]
                blobs_y = blobs_data[1]
                blobs_g = blobs_data[2]
                if coreSize == 'variable':
                    blobs_sigma = blobs_data[3]
            else:
                blobs_x = blobs_data[:,0]
                blobs_y = blobs_data[:,1]
                blobs_g = blobs_data[:,2]
                if coreSize == 'variable':
                    blobs_sigma = blobs_data[:,3]
            if coreSize == 'variable':
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                ax.scatter(blobs_x,blobs_y,c=blobs_g, s= blobs_sigma * 0.2 / np.min(blobs_sigma))
                plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                ax.scatter(blobs_x,blobs_y,c=blobs_g, s=0.2)
                plt.savefig("{}/blobs_{}_{}.png".format(plots_dir,case,timeStep), dpi=300, bbox_inches="tight")
                plt.close(fig)