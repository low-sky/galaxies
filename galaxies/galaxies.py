from astropy.coordinates import SkyCoord, Angle, FK5
from astroquery.ned import Ned
import astropy.units as u
from astropy.table import Table
import warnings
import numpy as np
import astropy

class Galaxy(object):
    def __init__(self, name):
        self.name = name
# An astropy coordinates structure
        self.coordinates = None
# This is the preferred name in a database.
        self.canonical_name = None
# With units
        self.distance = None
        self.inclination = None
        self.position_angle = None
        self.redshift = None
        self.vsys = None
        try:
            t = Ned.query_object(name)
            if len(t)==1:
                self.canonical_name = t['Object Name'][0]
                self.velocity = t['Velocity'][0]*u.km/u.s
                self.center_position = SkyCoord(t['RA(deg)'][0], t['DEC(deg)'][0],frame='fk5',
                                           unit='degree')
                self.redshift = t['Redshift'][0]
        except:
            warnings.warn("Unsuccessful query to NED")
            pass
        if name.upper() == 'M33':
            self.name = 'M33'
            self.distance = 8.4e5*u.pc
            self.center_position = SkyCoord(23.461667,30.660194,unit=(u.deg,u.deg),frame='fk5')
            self.position_angle = Angle(202*u.deg)
            self.inclination = Angle(56*u.deg)
            self.vsys = -179*u.km/u.s
        if name.upper() == 'M83':
            self.name = 'M83'
            self.distance = 4.8e6*u.pc
#            self.center_position = SkyCoord(23.461667,30.660194,unit=(u.deg,u.deg),frame='fk5')
            self.position_angle = Angle(225*u.deg)
            self.inclination = Angle(24*u.deg)
            self.vsys = 514*u.km/u.s
        if name.upper() == 'M100':
            self.name = 'M100'
            self.distance = 14.3e6*u.pc
#            self.center_position = SkyCoord(23.461667,30.660194,unit=(u.deg,u.deg),frame='fk5')
            self.position_angle = Angle(153*u.deg)
            self.inclination = Angle(30*u.deg)
            self.vsys = 1575*u.km/u.s
        if name.upper() == 'M64':
            self.name = 'M64'
            self.distance = 4.1e6*u.pc
            self.position_angle = Angle(-67.6*u.deg)
            self.inclination = Angle(58.9*u.deg)
            self.vsys = 411.3*u.km/u.s

    def __repr__(self):
        return "Galaxy {0} at RA={1}, DEC={2}".format(self.name,self.center_position.ra,self.center_position.dec)

    def radius(self, skycoord = None, ra = None, dec = None, 
               header = None, returnXY = False):
        if skycoord:
            PAs = self.center_position.position_angle(skycoord)
        elif type(header) is astropy.io.fits.header.Header:
            from astropy.wcs import WCS
            w = WCS(header)
            ymat, xmat = np.indices((w.celestial._naxis2,w.celestial._naxis1))
            ramat, decmat = w.celestial.wcs_pix2world(xmat,ymat,0)
            Offsets = SkyCoord(ramat,decmat,unit=(u.deg,u.deg))
            PAs = self.center_position.position_angle(Offsets)
        elif (np.any(ra) and np.any(dec)):
            Offsets = SkyCoord(ra,dec,unit=(u.deg,u.deg))
            PAs = self.center_position.position_angle(Offsets)
        else:
            warnings.warn('You must specify either RA/DEC, a header or a skycoord')
        GalPA = PAs - self.position_angle
        GCDist = Offsets.separation(self.center_position)
    # Transform into galaxy plane
        Rplane = self.distance*np.tan(GCDist)
        Xplane = Rplane * np.cos(GalPA)
        Yplane = Rplane * np.sin(GalPA)
        Xgal = Xplane
        Ygal = Yplane / np.cos(self.inclination)
        Rgal = (Xgal**2+Ygal**2)**0.5
        if returnXY:
            return (Xgal,Ygal)
        else:
            return Rgal

# push or pull override table using astropy.table

# Check name equivalencies

# Throwaway function to start development.  


