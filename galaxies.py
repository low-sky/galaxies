import astropy.coordinates as coords
import astroquery as query
import astropy.units as u
from astropy.table import Table

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
        
    def echoes (self, x):
        return x
    
# push or pull override table using astropy.table

# Check name equivalencies

# Throwaway function to start development.  


