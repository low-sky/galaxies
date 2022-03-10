
'''
Functions for generating different region shapes
across an image.
'''

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.utils.console import ProgressBar
import astropy.units as u
from matplotlib.path import Path
from scipy import ndimage as nd

try:
    from shapely.geometry import Polygon, Point
    HAS_SHAPLEY = True
except ImportError:
    HAS_SHAPLEY = False


def calculate_hexagons(centerx, centery, startx, starty, endx, endy,
                       radius, return_shapely=True):
    """
    Calculate a grid of hexagon coordinates of the given radius
    given lower-left and upper-right coordinates
    Returns a list of lists containing 6 tuples of x, y point coordinates
    These can be used to construct valid regular hexagonal polygons

    Adapted from: https://gist.github.com/urschrei/17cf0be92ca90a244a91

    """

    # First build the grid of hexagonal centres, to ensure the
    # given centres always remain at the centre of one hexagon.

    # Create positions of x-rows
    w = 2 * radius
    del_cent_x = 0.75 * w
    x_centres = np.append(np.arange(centerx, startx, -del_cent_x)[::-1],
                          np.arange(centerx + del_cent_x, endx, del_cent_x))

    # Create positions of y-rows. These are offset by h/2 every 2nd
    # column
    h = np.sqrt(3) * radius
    # Evens set every 2 h
    y_centres_even_low = np.arange(centery, starty, -h)[::-1]
    y_centres_even_high = np.arange(centery + h, endy, h)
    y_centres_odd_low = np.arange(centery - h / 2., starty, -h)[::-1]
    y_centres_odd_high = np.arange(centery + h / 2., endy, h)

    y_centres_even = np.append(y_centres_even_low, y_centres_even_high)
    y_centres_odd = np.append(y_centres_odd_low, y_centres_odd_high)

    pts = []

    for even in y_centres_even:
        pts.extend([(even, x) for x in x_centres[::2]])

    for odd in y_centres_odd:
        pts.extend([(odd, x) for x in x_centres[1::2]])

    # With the centres, now get the vertices of each
    polygons = []

    delta_y = radius * np.sqrt(3) / 2
    delta_x = radius / 2

    for pt in pts:
        starty = pt[0]
        startx = pt[1]

        p1x = startx + radius
        p1y = starty

        p2x = startx + delta_x
        p2y = starty + delta_y

        p3x = startx - delta_x
        p3y = starty + delta_y

        p4y = starty
        p4x = startx - radius

        p5x = startx - delta_x
        p5y = starty - delta_y

        p6x = startx + delta_x
        p6y = starty - delta_y

        poly = [(p1y, p1x),
                (p2y, p2x),
                (p3y, p3x),
                (p4y, p4x),
                (p5y, p5x),
                (p6y, p6x),
                (p1y, p1x)]
        polygons.append(poly)

    if return_shapely:
        if not HAS_SHAPLEY:
            raise ImportError("shapley could not be imported. Set "
                              "`return_shapely=False`.")
        shapely_polygons = []
        for poly, pt in zip(polygons, pts):
            shapely_polygons.append([pt, Polygon(poly)])
        return shapely_polygons
    else:
        return pts, polygons


def region_mask(poly, spat_coord_map):

    posns_y = []
    posns_x = []

    # Define a grid based on the shape boundaries
    min_x, min_y = np.floor(poly.bounds[:2])
    max_x, max_y = np.ceil(poly.bounds[2:])

    # Make spatial coordinate maps

    lat_map, lon_map = spat_coord_map
    lat_map = lat_map.value
    lon_map = lon_map.value

    # Slice out regions within the max and min:
    lat_mask = np.logical_and(lat_map > min_y,
                              lat_map < max_y)
    lon_mask = np.logical_and(lon_map > min_x,
                              lon_map < max_x)

    valid_mask = np.logical_and(lat_mask, lon_mask)

    # Slice out the valid mask
    valid_slice = nd.find_objects(valid_mask)[0]

    lat_vals = lat_map[valid_slice].ravel()
    lon_vals = lon_map[valid_slice].ravel()

    points = np.vstack((lon_vals, lat_vals)).T

    poly_verts = [(x, y) for x, y in zip(*poly.exterior.coords.xy)]

    path = Path(poly_verts)

    grid = path.contains_points(points)
    grid = grid.reshape(valid_mask[valid_slice].shape)

    valid_posns = np.where(grid)

    posns_y = valid_posns[0] + valid_slice[0].start
    posns_x = valid_posns[1] + valid_slice[1].start

    posns = np.vstack([posns_y, posns_x])

    return posns


def make_regions(data, gal, region_type='hexagon', radius=1 * u.deg,
                 center='gal', check_isfinite=False,
                 no_pixel_posns=False):
    '''
    Generate a set of equal
    '''

    # Will need to use wcs.pixel_shape in the future. But right now
    # the celestial WCS keeps all of the shapes from the original WCS
    # shape = (wcs_celest._naxis2, wcs_celest._naxis1)

    radius = radius.to(u.deg)

    if check_isfinite:
        raise NotImplementedError("This isn't working correctly.")
        mask_slice = nd.find_objects(np.isfinite(data))[0]

        sliced_data = data[mask_slice].copy()
        lon_extrema = sliced_data.world_extrema[0]
        lat_extrema = sliced_data.world_extrema[1]

    else:
        lon_extrema = data.world_extrema[0]
        lat_extrema = data.world_extrema[1]

    if center == 'gal':
        center_posn = [gal.center_position.ra, gal.center_position.dec]
    else:
        # Take the mid point of the map
        center_posn = [0.5 * np.sum(lon_extrema), 0.5 * np.sum(lat_extrema)]

    shp_regions = calculate_hexagons(center_posn[1].value,
                                     center_posn[0].value,
                                     lat_extrema.min().value,
                                     lon_extrema.min().value,
                                     lat_extrema.max().value,
                                     lon_extrema.max().value,
                                     radius.value)

    if no_pixel_posns:
        return shp_regions

    # Now find the pixel positions in the data for those regions.

    posns = []

    # Make the spatial coordinate map
    spat_coord_map = data.spatial_coordinate_map

    for region in ProgressBar(shp_regions):

        posns.append(region_mask(region[1], spat_coord_map))

    # Lastly attach an Rgal and PAgal at the centre of each region
    Rgal = []
    PAgal = []

    for region in shp_regions:

        Rgal.append(gal.radius(skycoord=SkyCoord(region[0][0] * u.deg,
                                                 region[0][1] * u.deg))[0].value)

        PAgal.append(gal.position_angles(skycoord=SkyCoord(region[0][0] * u.deg,
                                                           region[0][1] * u.deg))[0].value)

    return shp_regions, posns, Rgal, PAgal


def apply_to_region(data, posns, func=np.nanmean):
    '''
    Apply an operation to the pixel pos'ns defined in a polygon.
    '''

    return func(data[posns])


def make_table_regions(shp_regions, nside=6):
    '''
    Given centre and vertices, save the polygon edges to a table
    '''

    from astropy.table import Table

    shape_pts = {"x_c": [], "y_c": []}
    for i in range(nside + 1):
        shape_pts["x_{}".format(i)] = []
        shape_pts["y_{}".format(i)] = []

    for region in shp_regions:
        shape_pts['x_c'].append(region[0][0])
        shape_pts['y_c'].append(region[0][1])

        for j, (x, y) in enumerate(zip(*region[1].exterior.coords.xy)):
            shape_pts['x_{}'.format(j)].append(x)
            shape_pts['y_{}'.format(j)].append(y)

    # Extract the edge points from the poly file
    tab = Table(shape_pts)

    return tab


def load_regions(tab, nside=6):
    '''
    Given centre and vertices, save the polygon edges to a table
    '''

    centers = [(x, y) for x, y in zip(tab['x_c'], tab['y_c'])]

    polys = []

    for i in range(len(centers)):

        pts = [(tab['x_{}'.format(j)][i], tab['y_{}'.format(j)][i])
               for j in range(nside + 1)]

        print(pts)

        poly = Polygon(pts[:-1])

        polys.append(poly)

    return list(zip(centers, polys))
