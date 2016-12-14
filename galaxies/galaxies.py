
from astropy.coordinates import SkyCoord, Angle, FK5
from astroquery.ned import Ned
import astropy.units as u
from astropy.table import Table, Column
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
                        self.redshift = t['Redshift'][0] * \
                            u.dimensionless_unscaled
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
                self.vsys = -179 * u.km / u.s
            elif name.upper() == 'M83':
                self.name = 'M83'
                self.distance = 4.8e6 * u.pc
                self.position_angle = Angle(225 * u.deg)
                self.inclination = Angle(24 * u.deg)
                self.vsys = 514 * u.km / u.s
            elif name.upper() == 'NGC4303':
                self.name = 'NGC4303'
                self.distance = 14.5 * u.Mpc
                self.position_angle = Angle(0 * u.deg)
                self.inclination = Angle(18 * u.deg)
                self.vsys = 1569 * u.km / u.s
            elif name.upper() == 'M100':
                self.name = 'M100'
                self.distance = 14.3e6 * u.pc
    #            self.center_position = SkyCoord(23.461667,30.660194,unit=(u.deg,u.deg),frame='fk5')
                self.position_angle = Angle(153 * u.deg)
                self.inclination = Angle(30 * u.deg)
                self.vsys = 1575 * u.km / u.s
            elif name.upper() == 'M64':
                self.name = 'M64'
                self.distance = 4.1e6 * u.pc
                self.position_angle = Angle(-67.6 * u.deg)
                self.inclination = Angle(58.9 * u.deg)
                self.vsys = 411.3 * u.km / u.s
            elif name.upper() == 'NGC1672':
                self.position_angle = Angle(170 * u.deg)
            elif name.upper() == 'NGC4535':
                self.position_angle = Angle(0 * u.deg)
            elif name.upper() == 'NGC5068':
                self.position_angle = Angle(110 * u.deg)

            else:
                raise ValueError("The information for galaxy {} could not be "
                                 "found.")

    def __repr__(self):
        return "Galaxy {0} at RA={1}, DEC={2}".format(self.name,
                                                      self.center_position.ra,
                                                      self.center_position.dec)

    def radius(self, skycoord=None, ra=None, dec=None,
               header=None, returnXY=False):
        if skycoord:
            PAs = self.center_position.position_angle(skycoord)
            Offsets = skycoord
        elif isinstance(header, fits.Header):
            w = WCS(header)
            ymat, xmat = np.indices((w.celestial._naxis2, w.celestial._naxis1))
            ramat, decmat = w.celestial.wcs_pix2world(xmat, ymat, 0)
            Offsets = SkyCoord(ramat, decmat, unit=(u.deg, u.deg))
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

    def to_table(self):
        '''
        Return an `~astropy.table.Table` with the galactic parameters.
        '''

        tab = Table()

        tab["Name"] = Column([self.name], unit=None)
        tab["Center Position"] = Column([self.center_position])

        tab["Distance"] = Column([self.distance.value],
                                 unit=self.distance.unit)
        tab["Inclination"] = Column([self.inclination.value],
                                    unit=self.inclination.unit)
        tab["PA"] = Column([self.position_angle.value],
                           unit=self.position_angle.unit)
        tab["Vsys"] = Column([self.vsys.value],
                             unit=self.vsys.unit)

        if self.redshift is not None:
            tab["Redshift"] = Column([self.redshift.value],
                                     unit=self.redshift.unit)
        self.canonical_name = None

        if self.canonical_name is not None:
            tab["Canonical Name"] = Column([self.canonical_name],
                                           unit=None)

        return tab

# push or pull override table using astropy.table

# Check name equivalencies

# Throwaway function to start development.
