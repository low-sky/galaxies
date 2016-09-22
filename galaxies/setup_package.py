from __future__ import absolute_import

import os

def get_package_data():
    paths = [os.path.join('data','gal_base.fits')]
    return {'galaxies': paths}
