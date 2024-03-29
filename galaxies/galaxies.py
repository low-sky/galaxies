from astropy.coordinates import SkyCoord, Angle, FK5
from astroquery.ned import Ned
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
import warnings
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.console import ProgressBar

import pandas as pd
from scipy import interpolate,optimize  # Added since scipy functions are used.

def parse_galtable(galobj,name):
    table_name = get_pkg_data_filename('data/gal_base.fits',
                                       package='galaxies')
    galtable = Table.read(table_name)
    hits = [x for x in galtable if name in x['ALIAS']]
    if len(hits)>0:
        if len(hits)>1:
            exact = np.zeros(len(hits),dtype=np.bool)
            for i, h in enumerate(hits):
                exact[i] = np.any([n.strip()==name for n in h['ALIAS'].split(';')])
            if np.sum(exact)==1:
                thisobj = hits[(np.where(exact))[0][0]]
            else:
                raise Exception("More than one exact match in galbase")
        else:
            thisobj = hits[0]
        galobj.name = thisobj['NAME'].strip()
        galobj.vsys = thisobj['VRAD_KMS'] * u.km / u.s
        galobj.center_position = SkyCoord(
            thisobj['RA_DEG'], thisobj['DEC_DEG'],frame='fk5',
            unit='degree')
        galobj.distance = thisobj['DIST_MPC'] * u.Mpc
        galobj.inclination = thisobj['INCL_DEG'] * u.deg
        galobj.position_angle = thisobj['POSANG_DEG'] * u.deg
        galobj.provenance = 'GalBase'
        return True


class Galaxy(object):
    '''

    Parameters
    ----------
    name : str
        Name of the galaxy.the
    params : dict, optional
        Optionally provide custom parameter values as a dictionary.
    '''
    def __init__(self, name, params=None):

        self.name = name
# An astropy coordinates structure
        self.center_position = None
# This is the preferred name in a database.
        self.canonical_name = None
# With units
        self.distance = None
        self.inclination = None
        self.position_angle = None
        self.redshift = None
        self.vsys = None
        self.PA_is_kinematic = False
        self.provenance = None

        if params is not None:
            if not isinstance(params, dict):
                raise TypeError("params must be a dictionary.")

            required_params = ["center_position", "distance", "inclination",
                               "position_angle", "vsys"]
            optional_params = ["canonical_name", "redshift"]

            keys = params.keys()
            for par in required_params:
                if par not in keys:
                    raise ValueError("params is missing the required key"
                                     " {}".format(par))
                setattr(self, par, params[par])

            for par in optional_params:
                if par in keys:
                    setattr(self, par, params[par])

        else:

            if not parse_galtable(self, name):
                try:
                    t = Ned.query_object(name)
                    if len(t) == 1:
                        self.canonical_name = t['Object Name'][0]
                        self.velocity = t['Velocity'][0] * u.km / u.s
                        self.center_position = \
                            SkyCoord(t['RA'][0], t['DEC'][0],
                                     frame='fk5',
                                     unit='degree')
                        self.redshift = t['Redshift'][0]
                        self.provenance = 'NED'
                except:
                    raise ValueError
                    warnings.warn("Unsuccessful query to NED")
                    pass
            if name.upper() == 'M51':
                self.name = 'M51'
                self.distance = 7.6 * u.Mpc
                self.distance = 8.36 * u.Mpc

                self.center_position = SkyCoord(
                     202.46954345703125, 47.19514846801758,
                    unit=(u.deg, u.deg), frame='fk5')
                self.position_angle=Angle(172.0 * u.deg)
                self.inclination = Angle(20 * u.deg)
                self.velocity = 463.0 * u.km / u.s
                self.provenance = 'Override'
            if name.upper() == 'M33':
                self.name = 'M33'
                self.distance = 8.4e5 * u.pc
                self.center_position = \
                    SkyCoord(23.4607, 30.6583, unit=(u.deg, u.deg),
                             frame='fk5')
                self.position_angle = Angle(201.12 * u.deg)
                self.PA_is_kinematic = True
                self.inclination = Angle(55.08 * u.deg)
                self.velocity = -179 * u.km / u.s
                self.provenance = 'Override'
            elif name.upper() == 'M83':
                self.name = 'M83'
                self.distance = 4.8e6 * u.pc
                self.position_angle = Angle(225 * u.deg)
                self.inclination = Angle(24 * u.deg)
                self.velocity = 514 * u.km / u.s
                self.provenance = 'Override'
            elif name.upper() == 'NGC4303':
                self.name = 'NGC4303'
                self.distance = 11.6 * u.Mpc
                #self.position_angle = Angle(0 * u.deg)
                self.position_angle = Angle(313.2 * u.deg)  # PHANGS best fit parameters
                self.inclination = Angle(18 * u.deg)
                self.velocity = 1569 * u.km / u.s
                self.provenance = 'Override'
            elif name.upper() == 'M100':
                self.name = 'M100'
                self.distance = 14.3e6 * u.pc
                #self.center_position = SkyCoord(23.461667,30.660194,unit=(u.deg,u.deg),frame='fk5')
                self.position_angle = Angle(153 * u.deg)
                self.inclination = Angle(30 * u.deg)
                self.velocity = 1575 * u.km / u.s
                self.provenance = 'Override'
            elif name.upper() == 'M64':
                self.name = 'M64'
                self.distance = 4.1e6 * u.pc
                self.position_angle = Angle(-67.6 * u.deg)
                self.inclination = Angle(58.9 * u.deg)
                self.velocity = 411.3 * u.km / u.s
                self.provenance = 'Override'
            elif name.upper() == 'NGC4535':
                self.name = 'NGC4535'
                self.position_angle = Angle(0 * u.deg)
                self.provenance = 'Override'
            elif name.upper() == 'NGC5068':
                self.name = 'NGC5068'
                self.position_angle = Angle(110 * u.deg)
                self.provenance = 'Override'
            elif name.upper() == 'IC5332':
                self.position_angle = Angle(0 * u.deg)
                self.provenance = 'Override'
            # PHANGS Overrides from Sun et al. 2018
            elif name.upper() == 'NGC0628':
                self.distance = 9.0 * u.Mpc
                self.provenance = 'Override'
            elif name.upper() == 'NGC1672':
                self.name = 'NGC1672'
                self.position_angle = Angle(141.9 * u.deg)  # PHANGS best fit parameters
                self.inclination = Angle(24.7 * u.deg)      # PHANGS best fit parameters
                #self.position_angle = Angle(124. * u.deg) #http://iopscience.iop.org/article/10.1086/306781/pdf
                self.distance = 11.9 * u.Mpc
                #http://iopscience.iop.org/article/10.1086/306781/pdf
                #self.position_angle = Angle(170 * u.deg)
                self.provenance = 'Override'
            elif name.upper() == 'NGC2835':
                self.distance == 10.1 * u.Mpc
                self.provenance = 'Override'
            elif name.upper() == 'NGC3351':
                self.distance == 10.0 * u.Mpc
                self.provenance = 'Override'
            elif name.upper() == 'NGC3627':
                self.distance == 8.28 * u.Mpc
                self.provenance = 'Override'
            elif name.upper() == 'NGC4254':
                self.distance == 16.8 * u.Mpc
                self.provenance = 'Override'
            elif name.upper() == 'NGC4303':
                self.provenance = 'Override'
                self.distance == 17.6 * u.Mpc
            elif name.upper() == 'NGC4321':
                self.provenance = 'Override'
                self.distance == 15.2 * u.Mpc
            elif name.upper() == 'NGC4535':
                self.name = 'NGC4535'
                #self.position_angle = Angle(0 * u.deg)
                self.position_angle = Angle(179.4 * u.deg)  # PHANGS best fit parameters
                self.provenance = 'Override'
                self.distance == 15.8 * u.Mpc
            elif name.upper() == 'NGC5068':
                self.name = 'NGC5068'
                #self.position_angle = Angle(110 * u.deg)
                self.position_angle = Angle(352.5 * u.deg)  # PHANGS best fit parameters
                self.provenance = 'Override'
            elif name.upper() == 'NGC2835':
                self.name = 'NGC2835'
                self.position_angle = Angle(0.0 * u.deg)  # PHANGS best fit parameters
                self.provenance = 'Override'
            elif name.upper() == 'IC5332':
                self.name = 'IC5332'
                #self.position_angle = Angle(0 * u.deg)
                self.provenance = 'Override'
                self.distance == 9.0 * u.Mpc
                self.position_angle = Angle(110 * u.deg)
            elif name.upper() == 'NGC6744':
                self.provenance = 'Override'
                self.distance == 11.6 * u.Mpc

            if self.provenance is None:
                raise ValueError("The information for galaxy {}".format(name)+
                                 "could not be found.")

    def __repr__(self):
        return "Galaxy {0} at RA={1}, DEC={2}".format(self.name,
                                                      self.center_position.ra,
                                                      self.center_position.dec)

    def skycoord_grid(self, header=None, wcs=None):
        '''
        Return a grid of RA and Dec values.
        '''
        if header is not None:
            w = WCS(header)
        elif wcs is not None:
            w = wcs
        else:
            raise ValueError("header or wcs must be given.")
        # w = WCS(header)
        try:
            # This is deprecated
            ymat, xmat = np.indices((w.celestial._naxis2,
                                     w.celestial._naxis1))
        except AttributeError:
            ymat, xmat = np.indices(w.celestial.array_shape)

        ramat, decmat = w.celestial.wcs_pix2world(xmat, ymat, 0)
        return SkyCoord(ramat, decmat, unit=(u.deg, u.deg))

    def radius(self, skycoord=None, ra=None, dec=None,
               header=None, returnXY=False):
        if skycoord:
            PAs = self.center_position.position_angle(skycoord)
            Offsets = skycoord
        elif isinstance(header, fits.Header):
            Offsets = self.skycoord_grid(header=header)
            PAs = self.center_position.position_angle(Offsets)
        elif np.any(ra) and np.any(dec):
            Offsets = SkyCoord(ra, dec, unit=(u.deg, u.deg))
            PAs = self.center_position.position_angle(Offsets)
        else:
            warnings.warn('You must specify either RA/DEC, a header or a '
                          'skycoord')
        GalPA = PAs - self.position_angle
        GCDist = Offsets.separation(self.center_position)
        # Transform into galaxy plane
        Rplane = self.distance * np.tan(GCDist)
        Xplane = Rplane * np.cos(GalPA)
        Yplane = Rplane * np.sin(GalPA)
        Xgal = Xplane
        Ygal = Yplane / np.cos(self.inclination)
        Rgal = np.sqrt(Xgal**2 + Ygal**2)
        if returnXY:
            return (Xgal, Ygal)
        else:
            return Rgal

    def position_angles(self, skycoord=None, ra=None, dec=None,
                        header=None):
        X, Y = self.radius(skycoord=skycoord, ra=ra, dec=dec,
                           header=header, returnXY=True)

        return Angle(np.arctan2(Y, X))

    def to_center_position_pixel(self, wcs=None, header=None):

        if header is not None:
            wcs = WCS(header)

        if wcs is None:
            raise ValueError("Either wcs or header must be given.")

        return self.center_position.to_pixel(wcs)

    def rotcurve(self, mode='PHANGS',
                 #rcdir='/mnt/bigdata/PHANGS/OtherData/derived/Rotation_curves/'):
                 rcdir='/media/jnofech/BigData/galaxies/rotcurves/'):
        '''
        Reads a provided rotation curve table and
        returns interpolator functions for rotational
        velocity vs radius, and epicyclic frequency vs
        radius.

        Parameters:
        -----------
        mode : str
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

        # Basic info
        d = (self.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc

        rcdir_jnofech = '/media/jnofech/BigData/galaxies/rotcurves/'    # Joseph's main rotcurve directory,
                                                                        #   including the ones on the server.

        # Rotation Curves
        if mode.lower()=='phangs':
            if self.name.lower()=='m33':
                m33 = pd.read_csv('notphangsdata/m33_rad.out_fixed.csv')
                R = m33['r']
                vrot = m33['Vt']
                vrot_e = None
                print( "WARNING: M33 rotcurve error bars not accounted for!")
            else:
                fname = rcdir+(rcdir==rcdir_jnofech)*'phangs/'+self.name.lower()+"_co21_12m+7m+tp_RC.txt"
                R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
        elif mode.lower()=='diskfit12m':
            fname = rcdir+'diskfit12m/'+self.name.lower()+"_co21_12m+7m_RC.txt"    # Not on server.
            R, vrot, vrot_e = np.loadtxt(fname,skiprows=True,unpack=True)
        elif mode.lower()=='diskfit7m':
            fname = rcdir+'diskfit7m/'+self.name.lower()+"_co21_7m_RC.txt"         # Not on server.
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
#        if R[0]!=0:
#            R = np.roll(np.concatenate((R,[0]),0),1)
#            vrot = np.roll(np.concatenate((vrot,[0]),0),1)

        def bspline(X,Y,knots,k=3,lowclamp=False, highclamp=False):
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
            spl = interpolate.make_lsq_spline(X, Y, t, k,w)

            return spl
        # BSpline of vrot(R)
        K=3                # Order of the BSpline
        t,c,k = interpolate.splrep(R,vrot,s=0,k=K)
        vrot = interpolate.BSpline(t,c,k, extrapolate=False)     # Cubic interpolation of vrot(R).
                                                                # 'vrot' is now a function, not an array.
        # Creating "higher-resolution" rotation curve
        Nsteps = 10000
        R = np.linspace(R.min(),R.max(),Nsteps)

        return R, vrot, R_e, vrot_e

    def rotmap(self,header=None, position_angle=None, inclination=None):
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
        R : np.ndarray
            Map of radii in disk plane, up to
            extent of the rotcurve; in pc.
        Dec, RA : np.ndarray
            2D arrays of the ranges of Dec and
            RA (respectively), in degrees.
        '''
        # Basic info
        vsys = self.vsys
        if vsys==None:
            vsys = self.velocity
            # For some reason, some galaxies (M33, NGC4303...) have velocity listed as "velocity" instead of "vsys".
        if not inclination:
            I = self.inclination
        else:
            I = inclination

        if not position_angle:
            PA = (self.position_angle / u.deg * u.deg)        # Position angle (angle from N to line of nodes)
        else:
            PA = position_angle

        RA_cen = self.center_position.ra / u.deg * u.deg          # RA of center of galaxy, in degrees
        Dec_cen = self.center_position.dec / u.deg * u.deg        # Dec of center of galaxy, in degrees

                                                         # NOTE: The x-direction is defined as the LoN.
        d = (self.distance).to(u.parsec)                 # Distance to galaxy, from Mpc to pc

        # vrot Interpolation
        R_1d, vrot,R_e,vrot_e = self.rotcurve(mode=mode)

        # Generating displayable grids
        X,Y = self.radius(header=header, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope,
                                                      #    in Mpc.
        X = X.to(u.pc)
        Y = Y.to(u.pc)                               # Now they're in parsecs.
        # NOTE: - X is parallel to the line of nodes. The PA is simply angle from North to X-axis.
        #       - X- and Y-axes are exactly 90 degrees apart, which is only true for when X is parallel (or perp.)
        #               to the line of nodes.

        rad = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
        rad = ( (rad.value<R_1d.max()) * (rad.value>R_1d.min())).astype(int) * rad
        rad[ rad==0 ] = np.nan                         # Grid of radius, with values outside interpolation range removed.

        skycoord = self.skycoord_grid(header=header)     # Coordinates (RA,Dec) of the above grid at each point, in degrees.
        RA = skycoord.ra                             # Grid of RA in degrees.
        Dec = skycoord.dec                           # Grid of Dec in degrees.

        vobs = (vsys.value + vrot(R)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)
        return vobs, R, Dec, RA

# push or pull override table using astropy.table

# Check name equivalencies

# Throwaway function to start development.

