import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage, misc, interpolate, optimize
from scipy.interpolate import BSpline, make_lsq_spline

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.table import Table
from spectral_cube import SpectralCube
from galaxies import Galaxy

from pandas import DataFrame, read_csv
import pandas as pd
import statsmodels.formula.api as smf


def rotcurve(name,smooth='spline',knots=8):
    '''
    Reads a provided rotation curve table and
    returns interpolator functions for rotational
    velocity vs radius, and epicyclic frequency vs
    radius.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy that we care about.
    smooth : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (DEFAULT; uses specified # of knots)
        'brandt' (the analytical model)
    knots : int
        Number of internal knots in BSpline of
        vrot, which is used to calculate epicyclic
        frequency.
    mode : str
        'PHANGS'      - Uses PHANGS rotcurve.
        'DiskFit_12m' - Uses fitted rotcurve from
                        12m+7m data.        
        'DiskFit_7m'  - Uses fitted rotcurve from
                        7m data.
        
    Returns:
    --------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve, in km/s.
    k : scipy.interpolate.interp1d
        Function for the interpolated epicyclic
        frequency.
    '''
    
    # Basic info
    gal = Galaxy(name.upper())
    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc
    
    
    
    # Rotation Curves
    if mode.lower()=='phangs':
        if name=='m33':
            m33 = pd.read_csv('notphangsdata/m33_rad.out_fixed.csv')
            R = m33['r']
            vrot = m33['Vt']
            vrot_e = None
            print "WARNING: M33 rotcurve error bars not accounted for!"
        else:
            fname = "phangsdata/"+name.lower()+"_co21_12m+7m+tp_RC.txt"
            R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    elif mode.lower()=='diskfit_12m':
        fname = "diskfit/rotcurves/"+name.lower()+"_co21_12m+7m_RC.txt"
        R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    elif mode.lower()=='diskfit_7m':
        fname = "diskfit/rotcurves/"+name.lower()+"_co21_7m_RC.txt"
        R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    else:
        raise ValueError("'mode' must be PHANGS, diskfit_12m, or diskfit_7m!")
#         fname = "phangsdata/"+name.lower()+"_co21_12m+7m+tp_RC_OLD.txt"
#         R, vrot = np.loadtxt(fname,skiprows=True,unpack=True)
        # R = Radius from center of galaxy, in arcsec.
        # vrot = Rotational velocity, in km/s.
    # (!) When adding new galaxies, make sure R is in arcsec and vrot is in km/s, but both are 
    #     treated as unitless!
    
    # Units & conversions
    R = R*u.arcsec
    vrot = vrot*u.km/u.s
    R = R.to(u.rad)            # Radius, in radians.
    R = (R*d).value            # Radius, in pc, but treated as unitless.
    R_e = np.copy(R)           # Radius, corresponding to error bars.
    
    # Adding a (0,0) data point to rotation curve
    if R[0]!=0:
        R = np.roll(np.concatenate((R,[0]),0),1)
        vrot = np.roll(np.concatenate((vrot,[0]),0),1)
    
    
    
    # BSpline interpolation of vrot(R)
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,vrot,s=0,k=K)
    vrot = interpolate.BSpline(t,c,k, extrapolate=True)     # Cubic interpolation of vrot(R).
                                                            # 'vrot' is now a function, not an array.
    # Creating "higher-resolution" rotation curve
    Nsteps = 10000
    R = np.linspace(R.min(),R.max(),Nsteps)
    
    # SMOOTHING:
    if smooth==None or smooth.lower()=='none':
        print "WARNING: Smoothing disabled!"
    elif smooth.lower()=='spline':
        # BSpline of vrot(R)
        vrot = bspline(R,vrot(R),knots=knots,lowclamp=True)
    elif smooth.lower()=='brandt':
        def vcirc_brandt(r, *pars):
            '''
            Fit Eq. 5 from Meidt+08 (Eq. 1 Faber & Gallagher 79).
            This is taken right out of Eric Koch's code.
            '''
            n, vmax, rmax = pars
            numer = vmax * (r / rmax)
            denom = np.power((1 / 3.) + (2 / 3.) *\
                    np.power(r / rmax, n), (3 / (2 * n)))
            return numer / denom
        params, params_covariance = scipy.optimize.curve_fit(\
                                        vcirc_brandt,R_e,vrot(R_e),p0=(1,1,1),sigma=vrot_e,\
                                        bounds=((0.5,0,0),(np.inf,np.inf,np.inf)))
        print "n,vmax,rmax = "+str(params)
        vrot_b = vcirc_brandt(R,params[0],params[1],params[2])  # Array.

        # BSpline interpolation of vrot_s(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot_b,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=True)  # Now it's a function.
    elif smooth.lower()=='universal':
        def vcirc_universal(r, *pars):
            '''
            Fit Eq. 14 from Persic & Salucci 1995.
            '''
            v0, a, rmax = pars
            x = (r / rmax)
            return v0*np.sqrt( (0.72+0.44*np.log10(a))*(1.97*x**1.22)/(x**2 + 0.78**2)**1.43 +
                       1.6*np.exp(-0.4*a)*x**2/(x**2 + 1.5**2 *a**0.4) )
            
        params, params_covariance = optimize.curve_fit(\
                                        vcirc_universal,R_e,vrot(R_e),p0=(1,1,600),sigma=vrot_e,\
                                        bounds=((0,0.01,0),(np.inf,np.inf,np.inf)))
        print "v0,a,rmax = "+str(params)
        vrot_u = vcirc_universal(R,params[0],params[1],params[2])  # Array.

        # BSpline interpolation of vrot_u(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot_u,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=True)  # Now it's a function.
    
    # Epicyclic Frequency
    dVdR = np.gradient(vrot(R),R)
    k2 =  2.*(vrot(R)**2 / R**2 + vrot(R)/R*dVdR)
    k = interpolate.interp1d(R,np.sqrt(k2))
    
    
    return R, vrot, R_e, vrot_e, k     # NOTE: Here, 'vrot' may or may not be smoothed.
    

def rotmap(name,header=None):
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
    R : np.ndarray
        Map of radii of galaxy, in pc.
    Dec, RA : np.ndarray
        2D arrays of the ranges of Dec and 
        RA (respectively), in degrees.
    '''    
    # Basic info
    gal = Galaxy(name)
    vsys = gal.vsys
    if vsys==None:
        vsys = gal.velocity
        # For some reason, some galaxies (M33, NGC4303...) have velocity listed as "velocity" instead of "vsys".
    I = gal.inclination
    RA_cen = gal.center_position.ra / u.deg * u.deg          # RA of center of galaxy, in degrees 
    Dec_cen = gal.center_position.dec / u.deg * u.deg        # Dec of center of galaxy, in degrees
    PA = (gal.position_angle / u.deg * u.deg)        # Position angle (angle from N to line of nodes)
                                                     # NOTE: The x-direction is defined as the LoN.
    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc

    # vrot Interpolation
    R_1d, vrot, R_e, vrot_e, k_discard = rotcurve(name,smooth=None)  # Creates "vrot" interpolation function, and 1D array of R.


    # Generating displayable grids
    X,Y = gal.radius(header=header, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope, in Mpc.
    X = X.to(u.pc)
    Y = Y.to(u.pc)                               # Now they're in parsecs.
    # NOTE: - X is parallel to the line of nodes. The PA is simply angle from North to X-axis.
    #       - X- and Y-axes are exactly 90 degrees apart, which is only true for when X is parallel (or perp.)
    #               to the line of nodes.

    R = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
    R = (R.value<R_1d.max()).astype(int) * R  
    R[ R==0 ] = np.nan                           # Grid of radius, with values outside interpolation range removed.

    skycoord = gal.skycoord_grid(header=header)     # Coordinates (RA,Dec) of the above grid at each point, in degrees.
    RA = skycoord.ra                             # Grid of RA in degrees.
    Dec = skycoord.dec                           # Grid of Dec in degrees.


    vobs = (vsys.value + vrot(R)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)
    
    return vobs, R, Dec, RA

def bspline(X,Y,knots=8,k=3,lowclamp=False, highclamp=False):
    '''
    Returns a BSpline interpolation function
    of a provided 1D curve.
    With fewer knots, this will provide a
    smooth curve that ignores local wiggles.
    
    Parameters:
    -----------
    X,Y : np.ndarray
        1D arrays for the curve being interpolated.
    knots : int
        Number of INTERNAL knots, i.e. the number
        of breakpoints that are being considered
        when generating the BSpline.
    k : int
        Degree of the BSpline. Recommended to leave
        at 3.
    lowclamp : bool
        Enables or disables clamping at the lowest
        X-value.
    highclamp : bool
        Enables or disables clamping at the highest
        X-value.
        
    Returns:
    --------
    spl : scipy.interpolate._bsplines.BSpline
        Interpolation function that works over X's
        domain.
    '''
    
    # Creating the knots
    t_int = np.linspace(X.min(),X.max(),knots)  # Internal knots, incl. beginning and end points of domain.

    t_begin = np.linspace(X.min(),X.min(),k)
    t_end   = np.linspace(X.max(),X.max(),k)
    t = np.r_[t_begin,t_int,t_end]              # The entire knot vector.
    
    # Generating the spline
    w = np.zeros(X.shape)+1                     # Weights.
    if lowclamp==True:
        w[0]=X.max()*1000000                    # Setting a high weight for the X.min() term.
    if highclamp==True:
        w[-1]=X.max()*1000000                   # Setting a high weight for the X.max() term.
    spl = make_lsq_spline(X, Y, t, k,w)
    
    return spl

def localshear(name,smooth='spline',knots=8):
    '''
    Returns the local shear parameter (i.e. the
    Oort A constant) for a galaxy with a provided
    rotation curve, based on Equation 4 in Martin
    & Kennicutt (2001).
    
    Parameters:
    -----------
    name : str
        Name of the galaxy in question.
    smooth : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (DEFAULT; uses specified # of knots)
        'brandt' (the analytical model)
        'universal' (Persic & Salucci 1995)
    knots : int
        Number of INTERNAL knots in BSpline
        representation of rotation curve.
        
    Returns:
    --------
    A : scipy.interpolate._bsplines.BSpline
        Oort A "constant", as a function of 
        radius R, in km/s/kpc.
    '''
    gal = Galaxy(name)
    
    # Use "interp" to generate R, vrot (smoothed).
    R, vrot, R_e, vrot_e, k_discard = gal.rotcurve(smooth=smooth, knots=knots)
    
    # Oort A constant.
    Omega = vrot(R) / R     # Angular velocity.
    dOmegadR = np.gradient(Omega,R)
    A = (-1./2. * R*dOmegadR )*(u.kpc.to(u.pc)) # From km/s/pc to km/s/kpc.
    A = bspline(R[np.isfinite(A)],A[np.isfinite(A)],knots=999)
    
    return A

def linewidth_iso(name,beam,smooth='spline',knots=8,mode='PHANGS'):
    '''
    Returns the effective LoS velocity dispersion
    due to the galaxy's rotation, sigma_gal, for
    the isotropic case.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy in question.
    beam : float
        Beam width, in deg.
    smooth : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (DEFAULT; uses specified # of knots)
        'brandt' (the analytical model)
        'universal' (Persic & Salucci 1995)
    knots : int
        Number of INTERNAL knots in BSpline
        representation of rotation curve, which
        is used in calculation of epicyclic
        frequency (and, therefore, sigma_gal).    
    mode : str
        'PHANGS'      - Uses PHANGS rotcurve.
        'DiskFit_12m' - Uses fitted rotcurve from
                        12m+7m data.        
        'DiskFit_7m'  - Uses fitted rotcurve from
                        7m data.
        
    Returns:
    --------
    R : np.ndarray
        Radius array.
    sigma_gal : scipy.interpolate._bsplines.BSpline
        Interpolation function for sigma_gal that
        works over R.
    '''
    gal = Galaxy(name.upper())
    
    # Beam width
  
    beam = beam*u.deg.to(u.rad)                 # Beam size, in radians
    d = (gal.distance).to(u.pc)
    Rc = beam*d / u.rad                         # Beam size, in parsecs
    
    # Use "interp" to generate R, vrot (smoothed), k.
    R, vrot, R_e, vrot_e, k = gal.rotcurve(smooth=smooth,knots=knots,mode=mode)
    
    # Calculate sigma_gal = kappa*Rc
    sigma_gal = k(R)*Rc

    # Removing nans and infs
    # (Shouldn't be anything significant-- just a "nan" at R=0.)
    index = np.arange(sigma_gal.size)
    R_clean = np.delete(R, index[np.isnan(sigma_gal)==True])
    sigma_gal_clean = np.delete(sigma_gal, index[np.isnan(sigma_gal)==True])
    sigma_gal = bspline(R_clean,sigma_gal_clean,knots=20)


    # Cubic Interpolation of sigma_gal
    #K=3     # Order of the BSpline
    #t,c,k = interpolate.splrep(R,sigma_gal,s=0,k=K)
    #sigma_gal_spline = interpolate.BSpline(t,c,k, extrapolate=False)     # Cubic interpolation of sigma_gal(R).
    
    return R, sigma_gal

def moments(name,hdr,beam,I_mom0,I_tpeak,mode=''):
    '''
    Returns things like 'sigma' (line width, in km/s)
    or 'Sigma' (surface density) for a galaxy. The
    header, beam, and moment maps must be provided.
    
    Parameters:
    -----------
    name : str
        Name of the galaxy in question.
    hdr : astropy.io.fits.header.Header
        Header for the galaxy.
    beam : float
        Beam width, in deg.
    I_mom0 : np.ndarray
        0th moment, in K km/s.
    I_tpeak : np.ndarray
        Peak temperature, in K.

    Returns:
    --------
    R : np.ndarray
        Radius array.
    (s/S)igma : np.ndarray
        Maps for line width and surface density,
        respectively.
    '''

    gal = Galaxy(name.upper())
    rad = gal.radius(header=hdr)
    rad = rad.to(u.pc)                                      # Converts rad from Mpc to pc.
    d = gal.distance
    d = d.to(u.pc)                                          # Converts d from Mpc to pc.

    # Pixel sizes
    pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))*u.deg # The size of each pixel, in degrees. 
                                                                         # Ignore that third dimension; that's 
                                                                         # pixel size for the speed.
    #print "Pixel size, in degrees = "+str(pixsizes_deg[0])
    pixsizes = pixsizes_deg[0].to(u.rad)                    # Pixel size, in radians.
    pcperpixel =  pixsizes.value*d                          # Number of parsecs per pixel.
    #print "Pixel size, in parsecs = "+str(pcperpixel)
    pcperdeg = pcperpixel / pixsizes_deg[0]
    #print "There are +"+str(pcperdeg)+"."

    # Beam
    beam = beam * pcperdeg                                  # Beam size, in pc

    # Line width, Surface density
    alpha = 6.7  # (???) Units: (Msun pc^-2) / (K km s^-1)
    sigma = I_mom0 / (np.sqrt(2*np.pi) * I_tpeak)
    Sigma = alpha*I_mom0   # (???) Units: Msun pc^-2
    
    if mode=='sigma':
        return rad, sigma
    elif mode=='Sigma':
        return rad, Sigma
    else:
        print "SELECT A MODE."
