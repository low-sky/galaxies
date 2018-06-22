import numpy as np
import math

import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from galaxies.galaxies import Galaxy

from scipy import ndimage, misc, interpolate, optimize
from scipy.interpolate import BSpline, make_lsq_spline
from pandas import DataFrame, read_csv
import pandas as pd

# Import my own code
import galaxytools as tools


def rotcurve(gal,mode='PHANGS',
#              rcdir='/mnt/bigdata/PHANGS/OtherData/derived/Rotation_curves/'):
             rcdir='/media/jnofech/BigData/galaxies/rotcurves/'):
    '''
    Reads a provided rotation curve table and
    returns interpolator functions for rotational
    velocity vs radius, and epicyclic frequency vs
    radius.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    mode='PHANGS' : str
        'PHANGS'     - Uses PHANGS rotcurve.
        'diskfit12m' - Uses fitted rotcurve from
                        12m+7m data.        
        'diskfit7m'  - Uses fitted rotcurve from
                        7m data.
        
    Returns:
    --------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve, in km/s.
    R_e : np.ndarray
        1D array of original rotcurve radii, in pc.
    vrot_e : np.ndarray
        1D array of original rotcurve errors, in pc.
    '''
    
    # Do not include in galaxies.py!
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    

    d = (gal.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc   
    
    rcdir_jnofech = '/media/jnofech/BigData/galaxies/rotcurves/'     # Joseph's main rotcurve directory,
                                                                     #   including the ones on the server.
    # Rotation Curves
    if mode.lower()=='phangs':
        if gal.name.lower()=='m33':
            m33 = pd.read_csv('notphangsdata/m33_rad.out_fixed.csv')
            R = m33['r']
            vrot = m33['Vt']
            vrot_e = None
            print( "WARNING: M33 rotcurve error bars not accounted for!")
        else:
            fname = rcdir+(rcdir==rcdir_jnofech)*'phangs/'+name.lower()+"_co21_12m+7m+tp_RC.txt"
            R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    elif mode.lower()=='diskfit12m':
        fname = rcdir+'diskfit12m/'+name.lower()+"_co21_12m+7m_RC.txt"    # Not on server.
        R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    elif mode.lower()=='diskfit7m':
        fname = rcdir+'diskfit7m/'+name.lower()+"_co21_7m_RC.txt"         # Not on server.
        R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
    else:
        raise ValueError("'mode' must be PHANGS, diskfit12m, or diskfit7m!")

        # R = Radius from center of galaxy, in arcsec.
        # vrot = Rotational velocity, in km/s.
    # (!) When adding new galaxies, make sure R is in arcsec and vrot is in km/s, but both are 
    #     floats!
    
    # Units & conversions
    R = R*u.arcsec
    vrot = vrot*u.km/u.s
    R = R.to(u.rad)            # Radius, in radians.
    R = (R*d).value            # Radius, in pc, but treated as unitless.
    R_e = np.copy(R)           # Radius, corresponding to error bars.
    
    # Adding a (0,0) data point to rotation curve
#     if R[0]!=0:
#         R = np.roll(np.concatenate((R,[0]),0),1)
#         vrot = np.roll(np.concatenate((vrot,[0]),0),1)
    
    
    
    # BSpline interpolation of vrot(R)
    K=3                # Order of the BSpline
    t,c,k = interpolate.splrep(R,vrot,s=0,k=K)
    vrot = interpolate.BSpline(t,c,k, extrapolate=False)     # Cubic interpolation of vrot(R).
                                                            # 'vrot' is now a function, not an array.
    # Creating "higher-resolution" rotation curve
    Nsteps = 10000
    R = np.linspace(R.min(),R.max(),Nsteps)
    
    
    return R, vrot, R_e, vrot_e
    
def rotcurve_smooth(R,vrot,R_e,vrot_e=None,smooth='spline',knots=8):
    '''
    Takes a provided rotation curve
    and smooths it based on one of
    several models.
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated rotation
        curve, in km/s.
    R_e=None : np.ndarray
        1D array of original rotcurve radii, in pc.
    vrot_e=None : np.ndarray
        1D array of original rotcurve errors, in pc.
    smooth='spline' : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (DEFAULT; uses specified # of knots)
        'brandt' (an analytical model)
        'universal' (Persic & Salucci 1995)
    knots=8 : int
        Number of internal knots in BSpline of
        vrot, if mode=='spline'.
        
        
    Returns:
    --------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        SMOOTHED function for the interpolated
        rotation curve, in km/s.
    '''
    
    # SMOOTHING:
    if smooth==None or smooth.lower()=='none':
        print( "WARNING: Smoothing disabled!")
    elif smooth.lower()=='spline':
        # BSpline of vrot(R)
        vrot = bspline(R,vrot(R),knots=knots,lowclamp=False)
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
        params, params_covariance = optimize.curve_fit(\
                                        vcirc_brandt,R_e,vrot(R_e),p0=(1,1,1),sigma=vrot_e,\
                                        bounds=((0.5,0,0),(np.inf,np.inf,np.inf)))
        print( "n,vmax,rmax = "+str(params))
        vrot_b = vcirc_brandt(R,params[0],params[1],params[2])  # Array.

        # BSpline interpolation of vrot_b(R)
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
        print( "v0,a,rmax = "+str(params))
        vrot_u = vcirc_universal(R,params[0],params[1],params[2])  # Array.

        # BSpline interpolation of vrot_u(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot_u,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=True)  # Now it's a function.
    elif smooth.lower() in ['simple','exponential','expo']:
        def vcirc_simple(r, *pars):
            '''
            Fit Eq. 8 from Leroy et al. 2013.
            '''
            vflat, rflat = pars
            return vflat*(1.0-np.exp(-r / rflat))
            
        params, params_covariance = optimize.curve_fit(\
                                        vcirc_simple,R_e,vrot(R_e),p0=(1,1000),sigma=vrot_e,\
                                        bounds=((0,0.01),(np.inf,np.inf)))
        print( "vflat,rflat = "+str(params))
        vrot_s = vcirc_simple(R,params[0],params[1])  # Array.

        # BSpline interpolation of vrot_u(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot_s,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=True)  # Now it's a function.
    else:
        raise ValueError('Invalid smoothing mode.')
    
    return R, vrot

def epicycles(R,vrot):
    '''
    Returns the epicyclic frequency from a
    given rotation curve.
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated
        rotation curve, in km/s.
        Smoothing is recommended!
        
        
    Returns:
    --------
    k : scipy.interpolate.interp1d
        Function for the interpolated epicyclic
        frequency.
    '''
    # Epicyclic Frequency
    dVdR = np.gradient(vrot(R),R)
    k2 =  2.*(vrot(R)**2 / R**2 + vrot(R)/R*dVdR)
    k = interpolate.interp1d(R,np.sqrt(k2))
    
    return k
    

def rotmap(gal,header,mode='PHANGS'):
    '''
    Returns "observed velocity" map, and "radius
    map". (The latter is just to make sure that the
    code is working properly.)
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    header : astropy.io.fits.header.Header
        Header for the galaxy.
    mode='PHANGS' : str
        'PHANGS'     - Uses PHANGS rotcurve.
        'diskfit12m' - Uses fitted rotcurve from
                        12m+7m data.        
        'diskfit7m'  - Uses fitted rotcurve from
                        7m data.
        
    Returns:
    --------
    vobs : np.ndarray
        Map of observed velocity, in km/s.
    rad : np.ndarray
        Map of radii in disk plane, up to
        extent of the rotcurve; in pc.
    Dec, RA : np.ndarray
        2D arrays of the ranges of Dec and 
        RA (respectively), in degrees.
    '''    
    # Basic info
    
    # Do not include in galaxies.py!
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
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
    R_1d, vrot,R_e,vrot_e = rotcurve(name,mode=mode)  # Creates "vrot" interpolation function, and 1D array of R.


    # Generating displayable grids
    X,Y = gal.radius(header=header, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope, in Mpc.
    X = X.to(u.pc)
    Y = Y.to(u.pc)                               # Now they're in parsecs.
    # NOTE: - X is parallel to the line of nodes. The PA is simply angle from North to X-axis.
    #       - X- and Y-axes are exactly 90 degrees apart, which is only true for when X is parallel (or perp.)
    #               to the line of nodes.

    rad = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
    rad = ( (rad.value<R_1d.max()) * (rad.value>R_1d.min())).astype(int) * rad  
    rad[ rad==0 ] = np.nan                         # Grid of radius, with values outside interpolation range removed.

    skycoord = gal.skycoord_grid(header=header)     # Coordinates (RA,Dec) of the above grid at each point, in degrees.
    RA = skycoord.ra                             # Grid of RA in degrees.
    Dec = skycoord.dec                           # Grid of Dec in degrees.


    vobs = (vsys.value + vrot(rad)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)
    
    return vobs, rad, Dec, RA

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

def localshear(R,vrot):
    '''
    Returns the local shear parameter (i.e. the
    Oort A constant) for a galaxy with a provided
    rotation curve, based on Equation 4 in Martin
    & Kennicutt (2001).
    
    Parameters:
    -----------
    R : np.ndarray
        1D array of radii of galaxy, in pc.
    vrot : scipy.interpolate._bsplines.BSpline
        Function for the interpolated
        rotation curve, in km/s.
        Smoothing is recommended!
                        
    Returns:
    --------
    A : scipy.interpolate._bsplines.BSpline
        Oort A "constant", as a function of 
        radius R, in km/s/kpc.
    '''    
    # Oort A versus radius.
    Omega = vrot(R) / R     # Angular velocity.
    dOmegadR = np.gradient(Omega,R)
    A = (-1./2. * R*dOmegadR )*(u.kpc.to(u.pc)) # From km/s/pc to km/s/kpc.
    A = bspline(R[np.isfinite(A)],A[np.isfinite(A)],knots=999)
    
    return A

def linewidth_iso(gal,beam=None,smooth='spline',knots=8,mode='PHANGS'):
    '''
    Returns the effective LoS velocity dispersion
    due to the galaxy's rotation, sigma_gal, for
    the isotropic case.
    
    Parameters:
    -----------
    gal : str OR Galaxy
        Name of galaxy, OR Galaxy
        object.
    beam=None : float
        Beam width, in deg.
        Will be found automatically if not
        specified.
    smooth='spline' : str
        Determines smoothing for rotation curve.
        Available modes:
        'none'   (not recommended)
        'spline' (DEFAULT; uses specified # of knots)
        'brandt' (the analytical model)
        'universal' (Persic & Salucci 1995)
    knots=8 : int
        Number of INTERNAL knots in BSpline
        representation of rotation curve, which
        is used in calculation of epicyclic
        frequency (and, therefore, sigma_gal).    
    mode='PHANGS' : str
        'PHANGS'     - Uses PHANGS rotcurve.
        'diskfit12m' - Uses fitted rotcurve from
                        12m+7m data.        
        'diskfit7m'  - Uses fitted rotcurve from
                        7m data.
        
    Returns:
    --------
    sigma_gal : scipy.interpolate._bsplines.BSpline
        Interpolation function for sigma_gal that
        works over radius R.
    '''
    if isinstance(gal,Galaxy):
        name = gal.name.lower()
    elif isinstance(gal,str):
        name = gal.lower()
        gal = Galaxy(name.upper())
    else:
        raise ValueError("'gal' must be a str or galaxy!")
    
    # Beam width
    if beam==None:
        print('rc.linewidth_iso(): Beam size found automatically.')
        hdr = tools.hdr_get(gal)
        beam = hdr['BMAJ']
    beam = beam*u.deg.to(u.rad)                 # Beam size, in radians
    d = (gal.distance).to(u.pc)
    Rc = beam*d / u.rad                         # Beam size, in parsecs
    
    # Use "interp" to generate R, vrot (smoothed), k (epicyclic frequency).
    R, vrot, R_e, vrot_e = gal.rotcurve(mode=mode)
    k = epicycles(R,vrot)
    
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
    
    return sigma_gal


