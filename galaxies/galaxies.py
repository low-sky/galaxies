
from astropy.coordinates import SkyCoord, Angle, FK5
from astroquery.ned import Ned
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
import warnings
import numpy as np
from astropy.utils.data import get_pkg_data_filename

def parse_galtable(galobj,name):
    table_name = get_pkg_data_filename('data/gal_base.fits',
                                       package='galaxies')
    galtable = Table.read(table_name)
    hits = [x for x in galtable if name in x['ALIAS']]

    if len(hits)==1:
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
                            SkyCoord(t['RA(deg)'][0], t['DEC(deg)'][0],
                                     frame='fk5',
                                     unit='degree')
                        self.redshift = t['Redshift'][0]
                        self.provenance = 'NED'
                except:
                    warnings.warn("Unsuccessful query to NED")
                    pass

            if name.upper() == 'M33':
                self.name = 'M33'
                self.distance = 8.4e5 * u.pc
                self.center_position = \
                    SkyCoord(23.461667, 30.660194, unit=(u.deg, u.deg),
                             frame='fk5')
                self.position_angle = Angle(202 * u.deg)
                self.inclination = Angle(56 * u.deg)
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
                self.distance = 14.5 * u.Mpc
                self.position_angle = Angle(0 * u.deg)
                self.inclination = Angle(18 * u.deg)
                self.velocity = 1569 * u.km / u.s
                self.provenance = 'Override'
            elif name.upper() == 'M100':
                self.name = 'M100'
                self.distance = 14.3e6 * u.pc
    #            self.center_position = SkyCoord(23.461667,30.660194,unit=(u.deg,u.deg),frame='fk5')
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
            elif name.upper() == 'NGC1672':
                self.position_angle = Angle(170 * u.deg)
                self.provenance = 'Override'
            elif name.upper() == 'NGC4535':
                self.position_angle = Angle(0 * u.deg)
                self.provenance = 'Override'
            elif name.upper() == 'NGC5068':
                self.position_angle = Angle(110 * u.deg)
                self.provenance = 'Override'
#            else:

            if not self.provenance:
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
        w = WCS(header)
        ymat, xmat = np.indices((w.celestial._naxis2, w.celestial._naxis1))
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

    def interp(self, smoothing=False):
        '''
        Reads a provided rotation curve table from the
        "phangsdata" folder, and returns interpolator 
        functions for rotational velocity vs radius, 
        and epicyclic frequency vs radius.
        WARNING: Only for NGC1672 and M33 at the moment.

        Parameters:
        -----------
        smoothing : bool
            Enables or disables smoothing. (False by
            default. Enabling it doesn't seem to help 
            much.)

        Returns:
        --------
        R : np.ndarray
            1D array of radii of galaxy, in pc.
        vrot_i : scipy.interpolate.interpolate.interp1d
            Function for the interpolated rotation
            curve.
        k2_i : scipy.interpolate.interpolate.interp1d
            Function for the interpolated epicyclic
            frequency squared.
        '''
	from scipy import interpolate			# Had to be imported here, otherwise I'd
							# get a "global name 'interpolate' is not
							# defined" error message for some reason.

        # Basic info
        d = (self.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc

        # Rotation Curves
        if self.name.upper()=='NGC1672':
            fname = "phangsdata/NGC1672_co21_12m+7m+tp_RC.txt"
            hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
            R, vrot = np.loadtxt(fname,skiprows=True,unpack=True)
            # R = Radius from center of galaxy, in arcsec.
            # vrot = Rotational velocity, in km/s.
        elif self.name.upper()=='M33':
            m33 = pd.read_csv('phangsdata/m33_rad.out.csv')
            R = m33['r']
            vrot = m33['Vt']
        # (!) When adding new galaxies, make sure R is in arcsec and vrot is in km/s, but
        #     both are treated as unitless!

        # Adding a (0,0) data point to rotation curve
        if R[0]!=0:
            R = np.roll(np.concatenate((R,[0]),0),1)
            vrot = np.roll(np.concatenate((vrot,[0]),0),1)

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

        # Creating "higher-resolution" rotation curve
        Nsteps = 10000
        R = np.linspace(R.min(),R.max(),Nsteps)
        vrot = vrot_spline  # This is now a function, not an array.


        # Epicyclic Frequency
        dVdR = np.gradient(vrot(R),R)
        k2 =  2.*(vrot(R)**2 / R**2 + vrot(R)/R*dVdR)    # This is \kappa^2.


        # INCLUDE k IN TABLE
        tab = Table([R,vrot(R),k2], names=('R','vrot','k2'), \
                    meta={'name':'Radius, Rotational Velocity, and Epicyclic Frequency'})
        tab['R'].unit = 'pc'
        tab['vrot'].unit = 'km/s'
        tab['k2'].unit = '(km/s/pc)^2'

        vrot_i = interpolate.interp1d(tab['R'],tab['vrot'])      # The function "vrot_i" interpolates values of 
                                                                 #     vrot at some radius.
        k2_i = interpolate.interp1d(tab['R'],tab['k2'])          # The function "k2_i" interpolates values of 
                                                                 #     \kappa^2 at some radius.

        return R, vrot_i, k2_i

    def rotmap(self):
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
            Maps of Dec and RA (respectively), 
            in degrees.
        '''    
        # Basic info
        vsys = self.vsys
        if self.name.upper()=='M33':
            vsys = self.velocity
            # For some reason, M33's "Galaxy" object has velocity listed as "velocity" instead of "vsys".
        I = self.inclination
        RA_cen = self.center_position.ra / u.deg * u.deg          # RA of center of galaxy, in degrees 
        Dec_cen = self.center_position.dec / u.deg * u.deg        # Dec of center of galaxy, in degrees
        PA = (self.position_angle / u.deg * u.deg)        # Position angle (angle from N to line of nodes)
                                                         # NOTE: The x-direction is defined as the LoN.
        d = (self.distance).to(u.parsec)                  # Distance to galaxy, from Mpc to pc
        

        # Header
        if self.name.upper()=='NGC1672':
            hdr = fits.getheader('phangsdata/ngc1672_co21_12m+7m+tp_mom0.fits')
        elif self.name.upper()=='M33':
            hdr = fits.getheader\
            ('phangsdata/M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.peakvels.fits')

        # vrot Interpolation
        R_1d, vrot, k2_discard = self.interp(smoothing=0)  # Creates "vrot" interpolation function,
                                                           #    and 1D array of R.


        # Generating displayable grids
        X,Y = self.radius(header=hdr, returnXY=True)  # Coordinate grid in galaxy plane, as "seen" by telescope,
                                                      #    in Mpc.
        X = X.to(u.pc)
        Y = Y.to(u.pc)                               # Now they're in parsecs.
        # NOTE: - X is parallel to the line of nodes. The PA is simply angle from North to X-axis.
        #       - X- and Y-axes are exactly 90 degrees apart, which is only true for when X is parallel (or perp.)
        #               to the line of nodes.

        R = np.sqrt(X**2 + Y**2)                     # Grid of radius in parsecs.
        R = (R.value<R_1d.max()).astype(int) * R  
        R[ R==0 ] = np.nan                           # Grid of radius, with values outside interpolation range
                                                     #    removed.

        skycoord = self.skycoord_grid(header=hdr)   # Coordinates (RA,Dec) of the above grid at each point, 
                                                    #    in degrees.
        RA = skycoord.ra                             # Grid of RA in degrees.
        Dec = skycoord.dec                           # Grid of Dec in degrees.


        vobs = (vsys.value + vrot(R)*np.sin(I)*np.cos( np.arctan2(Y,X) )) * (u.km/u.s)

        return vobs, R, Dec, RA

# push or pull override table using astropy.table

# Check name equivalencies

# Throwaway function to start development.
