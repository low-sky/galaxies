
'''
Functions for generating different region shapes
across an image.
'''

import numpy as np

try:
    from shapely.geometry import Polygon, Point
    HAS_SHAPLEY = True
except ImportError:
    HAS_SHAPLEY = False


def calculate_hexagons(centerx, centery, shapex, shapey, radius,
                       return_shapely=True):
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
    x_centres = np.append(np.arange(centerx, 0., -del_cent_x)[::-1],
                          np.arange(centerx + del_cent_x, shapex, del_cent_x))

    # Create positions of y-rows. These are offset by h/2 every 2nd
    # column
    h = np.sqrt(3) * radius
    # Evens set every 2 h
    y_centres_even_low = np.arange(centery, 0., -h)[::-1]
    y_centres_even_high = np.arange(centery + h, shapey, h)
    y_centres_odd_low = np.arange(centery - h / 2., 0., -h)[::-1]
    y_centres_odd_high = np.arange(centery + h / 2., shapey, h)

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
        for poly in polygons:
            shapely_polygons.append(Polygon(poly))
        return shapely_polygons
    else:
        return pts, polygons


def region_mask(poly, shape, return_mask=True, return_posns=True):

    if return_mask:
        mask = np.zeros(shape, dtype=bool)

    if return_posns:
        posns_y = []
        posns_x = []

    # Define a grid based on the shape boundaries
    min_x, min_y = np.floor(poly.bounds[:2]).astype(int) - 1
    max_x, max_y = np.ceil(poly.bounds[2:]).astype(int)

    yy, xx = np.mgrid[max(min_y, 0):min(max_y, shape[0]),
                      max(min_x, 0):min(max_x, shape[1])]

    for y, x in zip(yy.ravel(), xx.ravel()):

        # Define pixels based on the mid-point so they have an area of 1.
        pix_region = Polygon([(x - 0.5, y - 0.5),
                              (x + 0.5, y - 0.5),
                              (x + 0.5, y + 0.5),
                              (x - 0.5, y + 0.5)])

        if poly.intersects(pix_region):
            if return_mask:
                mask[y, x] = True
            if return_posns:
                posns_y.append(y)
                posns_x.append(x)

    return_both = return_mask * return_posns

    if return_posns:
        posns = np.vstack([posns_y, posns_x])

    if return_both:
        return mask, posns
    elif return_mask:
        return mask
    elif return_posns:
        return posns
