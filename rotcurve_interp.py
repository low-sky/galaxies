import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage, misc, interpolate
from scipy.interpolate import BSpline

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.table import Table
from spectral_cube import SpectralCube
from galaxies import Galaxy

from pandas import DataFrame, read_csv
import pandas as pd
import statsmodels.formula.api as smf


def interp(name,smoothing=False):
    '''
    Reads a provided rotation curve table and
    returns interpolator functions for rotational
    velocity vs radius, and epicyclic frequency vs
    radius.
    WARNING: Only for NGC1672 and M33 at the moment.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy that we care about.
        
    Returns:
    --------
    R : np.ndarray
        Radii of galaxy, in pc.
    vrot_i : scipy.interpolate.interpolate.interp1d
        Function for the interpolated rotation
        curve.
    k2_i : scipy.interpolate.interpolate.interp1d
        Function for the interpolated epicyclic
        frequency squared.
    '''
    if name=='NGC1672':
        fname = "phangsdata/NGC1672_co21_12m+7m+tp_RC.txt"
        hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
    # Basic info
    gal = Galaxy(name)
    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc
    
    if name=='NGC1672':
        #(!) WARNING - This works for the very, very specific case of where the text file is just two columns,
        #     exactly as described below. Not quite sure how to get it working for the general case yet.
        R, vrot = np.loadtxt(fname,skiprows=True,unpack=True)
        # R = Radius from center of galaxy, in arcsec.
        # vrot = Rotational velocity, in km/s.
    elif name=='M33':
        m33 = pd.read_csv('phangsdata/m33_rad.out.csv')
        R = m33['r']
        vrot = m33['Vt']
    # (!) When adding new galaxies, make sure R is in arcsec and vrot is in km/s, but both are treated as unitless!
    
    if R[0]!=0:
        R = np.roll(np.concatenate((R,[0]),0),1)
        vrot = np.roll(np.concatenate((vrot,[0]),0),1)
        # Adds a (0,0) data point.
    
    R = R*u.arcsec
    vrot = vrot*u.km/u.s
        
    # Unit conversions
    R = R.to(u.rad)            # Radius, in radians.
    R = (R*d).value            # Radius, in pc, but treated as unitless.
    
    # Cubic Interpolation of vrot(R)
    K=3                # Order of the BSpline
    if smoothing==True:
        m = R.size
        s = m-np.sqrt(2*m) #(https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html)
        t,c,k = interpolate.splrep(R,vrot,s=s,k=K)
    else:
        t,c,k = interpolate.splrep(R,vrot,s=0,k=K)
    vrot_spline = interpolate.BSpline(t,c,k, extrapolate=True)     # Cubic interpolation of vrot(R).
    
    # "Redefining" things
    Nsteps = 10000
    R = np.linspace(R.min(),R.max(),Nsteps)
    vrot = vrot_spline  # This is now a function, not an array.
    
    
    # Epicyclic Frequency
    dVdR = np.gradient(vrot(R),R)
    k2 =  2.*(vrot(R)**2 / R**2 + vrot(R)/R*dVdR)    # This is \kappa^2.
    
    
    # INCLUDE k IN TABLE
    tab = Table([R,vrot(R),k2], names=('R','vrot','k2'), meta={'name':'Radius, Rotational Velocity, and Epicyclic Frequency'})
    tab['R'].unit = 'pc'
    tab['vrot'].unit = 'km/s'
    tab['k2'].unit = '(km/s/pc)^2'
    
    vrot_i = interpolate.interp1d(tab['R'],tab['vrot'])      # The function "y" interpolates values of vrot at some radius.
    k2_i = interpolate.interp1d(tab['R'],tab['k2'])          # The function "y" interpolates values of vrot at some radius.

    return R, vrot_i, k2_i


def rotmap(name):
    '''
    Returns "observed velocity" map, and "rotation
    map". (The latter is just to make sure that the
    code is working properly.)
    WARNING: Only works for NGC1672 at the moment.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy that we care about.
        
    Returns:
    --------
    vobs : np.ndarray
        Map of observed velocity, in km/s.
    Radi : np.ndarray
        Map of radii of galaxy, in pc.
    Dec, RA : np.ndarray
        1D arrays of the ranges of Dec and 
        RA (respectively), in degrees.
    '''    
    # Basic info
    gal = Galaxy(name)
    vsys = gal.vsys
    if name=='M33':
        vsys = -179.*u.km/u.s
        # For some godforsaken reason, M33's "Galaxy" object does not have a systemic velocity.
    I = gal.inclination
    RA_cen = gal.center_position.ra / u.deg * u.deg          # RA of center of galaxy, in degrees 
    Dec_cen = gal.center_position.dec / u.deg * u.deg        # Dec of center of galaxy, in degrees
    PA = (gal.position_angle / u.deg * u.deg)        # Position angle (angle from N to line of nodes)
                                                     # NOTE: The x-direction is defined as the LoN.
    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc

    # Fixing the Position Angle (if necessary):
    if name=='NGC1672':
        print "Provided PA is "+str(gal.position_angle)+"degrees CCW from North."
        #gal.position_angle = gal.position_angle + 180*u.deg
        print "Modified PA is "+str(gal.position_angle)+"degrees CCW from North."
    elif name=='M33':
        print "It's M33, so the PA doesn't need fixing."


    # Header
    if name=='NGC1672':
        hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
    elif name=='M33':
        hdr = fits.getheader('phangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')

    # vrot Interpolation
    R_1d, vrot, k2_discard = interp(name,smoothing=0)  # Creates "vrot" interpolation function, and 1D array of R.


    # Generating displayable grids
    X,Y = gal.radius(header=hdr, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope, in Mpc.
    X = X.to(u.pc)
    Y = Y.to(u.pc)                               # Now they're in parsecs.
    # NOTE: - X is parallel to the line of nodes. The PA is simply angle from North to X-axis.
    #       - X- and Y-axes are exactly 90 degrees apart, which is only true for when X is parallel (or perp.)
    #               to the line of nodes.

    R = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
    R = (R.value<R_1d.max()).astype(int) * R  
    R[ R==0 ] = np.nan                           # Grid of radius, with values outside interpolation range removed.

    skycoord = gal.skycoord_grid(header=hdr)     # Coordinates (RA,Dec) of the above grid at each point, in degrees.
    RA = skycoord.ra                             # Grid of RA in degrees.
    Dec = skycoord.dec                           # Grid of Dec in degrees.


    vobs = (vsys.value + vrot(R)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)
    
    return vobs, R, Dec, RA
