from astropy.coordinates import ICRS
from astroquery.ned import Ned
import astropy.units as u
from astropy.table import Table
import warnings

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
        self.velocity = None
        try:
            t = Ned.query_object(name)
            if len(t)==1:
                self.canonical_name = t['Object Name'][0]
                self.velocity = t['Velocity'][0]*u.km/u.s
                self.coordinates = ICRS(ra=t['RA(deg)'][0],
                                        dec=t['DEC(deg)'][0],
                                        unit=(u.degree,u.degree))
                self.redshift = t['Redshift'][0]
        except:
            warnings.warn("Unsuccessful query to NED")
            pass



        
    def echoes (self, x):
        return x
    
# push or pull override table using astropy.table

# Check name equivalencies

# Throwaway function to start development.  


