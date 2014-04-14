import astropy.config
from astropy.coordinates import ICRS
from astroquery.ned import Ned
import astropy.units as u
from astropy.table import Table
import warnings
import os

class Galaxy(object):

####################
# Constructor
####################

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

# overrideDB will point to the file that overrides current values in the table.
        self.overrideDB = self._overrideDB_name_default()
# galaxyDB points to a cached database
        self.galaxyDB = self._galaxyDB_name_default()
# Provenance of entries?
        self.data_source = None
        
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

        try:
            DB = read_overrideDB()
        except:
            DB = self._initDB()

# Should these be private methods?

    def read_overrideDB(self, DBname):
        if DBname is None:
            return Table.read(self.overrideDB,format='csv')
        else:
            return Table.read(DBname,format='csv')
        return
    
    def write_overrideDB(self, galaxyDB, DBname=None):
        try:
            if DBname is None:
                galaxyDB.writeto(self.overrideDB,format='csv')
            else:
                galaxyDB.writeto(DBname,format='csv')
        except:
            warnings.warn('No database written.')
        return

    def override(self, keyword, value):
        return

    def printDB(self,galaxyDB):
        return galaxyDB

###########
#  Prive
###########

    def _initDB(self):
        fields = []
        dtype = []
        fields.append('name')
        dtype.append('S24')
        fields.append('canonical_name')
        dtype.append('S24')
        fields.append('distance_mpc')
        dtype.append('f8')
        fields.append('inclination_deg')
        dtype.append('f8')
        fields.append('position_angle_deg')
        dtype.append('f8')
        fields.append('velocity_kms')
        dtype.append('f8')
        fields.append('redshift')
        dtype.append('f8')        
        galaxyDB = Table(names=fields,dtype=dtype,masked=True)
        return galaxyDB


    def _galaxyDB_name_default(self):
        astropy_dir =os.path.dirname(astropy.config.paths.get_config_dir())
        galaxies_dir = astropy_dir+'/galaxies/'
        if not os.path.isdir(galaxies_dir):
            try:
                os.mkdir(galaxies_dir)
            except:
                warnings.warn('Could not create galaxies directory in .astropy folder')
        DBname = galaxies_dir+'galaxiesDB.csv'
        return DBname


    def _overrideDB_name_default(self):
        astropy_dir =os.path.dirname(astropy.config.paths.get_config_dir())
        galaxies_dir = astropy_dir+'/galaxies/'
        if not os.path.isdir(galaxies_dir):
            try:
                os.mkdir(galaxies_dir)
            except:
                warnings.warn('Could not create galaxies directory in .astropy folder')
        DBname = galaxies_dir+'overrideDB.csv'
        return DBname

# push or pull override table using astropy.table

# Check name equivalencies

# Throwaway function to start development.  


