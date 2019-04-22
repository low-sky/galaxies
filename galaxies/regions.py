
'''
Functions for generating different region shapes
across an image.
'''

import numpy as np


def calculate_hexagons(centerx, centery, shapex, shapey, radius):
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

    # delta_x = radius * np.sqrt(3) / 2
    # delta_y = radius / 2

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

        poly = [
            (p1y, p1x),
            (p2y, p2x),
            (p3y, p3x),
            (p4y, p4x),
            (p5y, p5x),
            (p6y, p6x),
            (p1y, p1x)]
        polygons.append(poly)

    return pts, polygons
