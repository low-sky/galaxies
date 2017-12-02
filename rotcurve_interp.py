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

def interp(fname,smoothing=False):
    '''
    Reads a provided rotation curve table and
    returns interpolator functions for rotational
    velocity vs radius, and epicyclic frequency vs
    radius.
    WARNING: Only works for NGC1672 at the moment.
    
    Parameters:
    -----------
    fname : str
        Name of the text file to be read, including
        '.txt' suffix.
        
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
    d = 12.3           # Distance, in Mpc.
    d = d*1000000      # Distance, in pc.
    
    arcsec_to_deg = 1/3600.  # degree / arcsec
    deg_to_rad = np.pi/180.  # rad / degree
    arcsec_to_rad = arcsec_to_deg * deg_to_rad  # rad / arcsec
    
    
    #(!) WARNING - This works for the very, very specific case of where the text file is just two columns, exactly as described below. Not quite sure how to get it working for the general case yet.
    R, vrot = np.loadtxt(fname,skiprows=True,unpack=True)
    # R = Radius from center of galaxy, in arcsec.
    # vrot = Rotational velocity, in km/s.
    
    if R[0]!=0:
        R = np.roll(np.concatenate((R,[0]),0),1)
        vrot = np.roll(np.concatenate((vrot,[0]),0),1)
        # Adds a (0,0) data point.
        
        
    # Unit conversions
    R = R*arcsec_to_rad  # Radius, in radians.
    R = R*d              # Radius, in pc.
    
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


def rotmap(name='NGC1672'):
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
    Radius : np.ndarray
        Map of radii of galaxy, in pc.
    Dec, RA : np.ndarray
        1D arrays of the ranges of Dec and 
        RA (respectively), in degrees.
    '''    
    # Basic info
    gal = Galaxy(name)
    vsys = gal.vsys
    I = gal.inclination
    RA_cen = gal.center_position.ra / u.deg * u.deg          # RA of center of galaxy, in degrees 
    Dec_cen = gal.center_position.dec / u.deg * u.deg        # Dec of center of galaxy, in degrees
    theta_o = (gal.position_angle / u.deg * u.deg)        # Position angle (angle from "north" of line of nodes)
    d = (gal.distance).to(u.parsec)                     # Distance to galaxy, from Mpc to pc
    
    # Fixing the Position Angle (if necessary):
    if name=='NGC1672':
        gal.position_angle = gal.position_angle + 180*u.deg

    
    # Header, Interpolation Function
    if name=='NGC1672':
        fname = "phangsdata/NGC1672_co21_12m+7m+tp_RC.txt"
        hdr = fits.getheader('ngc1672_co21_12m+7m+tp_mom0.fits')
    R_discard, vrot, k2_discard = interp(fname,smoothing=0)  # Creates "vrot" interpolation function.

    
    # Generating displayable grids
    X,Y = gal.radius(header=hdr, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope, in Mpc.
    X = X.to(u.pc)
    Y = Y.to(u.pc)                               # Now they're in parsecs.
    
    R = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
    R = (R.value<R_discard.max()).astype(int) * R  
    R[ R==0 ] = np.nan                           # Grid of radius, with values outside interpolation range removed.
    
    skycoord = gal.skycoord_grid(header=hdr)     # Coordinates (RA,Dec) of the above grid at each point, in degrees.
    RA = skycoord.ra                             # Grid of RA in degrees.
    Dec = skycoord.dec                           # Grid of Dec in degrees.


    vobs = (vsys.value + vrot(R)*np.sin(I)*np.sin( np.arctan2(Y,X) )) * (u.km/u.s)
    
    return vobs, R, Dec, RA, vrot
