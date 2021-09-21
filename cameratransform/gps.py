#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gps.py

# Copyright (c) 2017-2021, Richard Gerum
#
# This file is part of the cameratransform package.
#
# cameratransform is free software: you can redistribute it and/or modify
# it under the terms of the MIT licence.
#
# cameratransform is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the license
# along with cameratransform. If not, see <https://opensource.org/licenses/MIT>

import numpy as np
import re


def formatGPS(lat, lon, format=None, asLatex=False):
    """
    Formats a latitude, longitude pair in degrees according to the format string.
    The format string can contain a %s, to denote the letter symbol (N, S, W, E) and up to three number formaters
    (%d or %f), to denote the degrees, minutes and seconds. To not lose precision, the last one can be float number.

    common formats are e.g.:

       +--------------------------------+--------------------------------------+
       | format                         | output                               |
       +================================+===================+==================+
       | %2d° %2d' %6.3f" %s (default)  | 70° 37'  4.980" S | 8°  9' 26.280" W |
       +--------------------------------+-------------------+------------------+
       | %2d° %2.3f' %s                 | 70° 37.083' S     | 8°  9.438' W     |
       +--------------------------------+-------------------+------------------+
       | %2.3f°                         | -70.618050°       | -8.157300°       |
       +--------------------------------+-------------------+------------------+

    Parameters
    ----------
    lat: number
        the latitude in degrees
    lon: number
        the longitude in degrees
    format: string
        the format string
    asLatex: bool
        whether to encode the degree symbol

    Returns
    -------
    lat: string
        the formatted latitude
    lon: string
        the formatted longitude

    Examples
    --------

    >>> import cameratransform as ct

    Convert a coordinate pair to a formatted string:

    >>> lat, lon = ct.formatGPS(-70.61805, -8.1573)
    >>> lat
    '70° 37\\'  4.980" S'
    >>> lon
    ' 8°  9\\' 26.280" W'

    or use a custom format:

    >>> lat, lon = ct.formatGPS(-70.61805, -8.1573, format="%2d° %2.3f %s")
    >>> lat
    '70° 37.083 S'
    >>> lon
    ' 8° 9.438 W'

    """
    import re
    # default format
    if format is None:
        format = "%2d° %2d' %6.3f\" %s"
    # try to split the format string into it's place holders
    match = re.findall(r"(%[.\d]*[sdf])", format)
    if len(match) == 0:
        raise ValueError("no valid format place holder specified")

    # try to find a %s to see if we have to use a letter symbol or a negative sign
    use_letter = False
    counter = 0
    fmt_degs = None
    fmt_mins = None
    fmt_sec = None
    for entry in match:
        if entry[-1] == "s":
            use_letter = True
        else:
            # store the formats of the degrees, minutes and seconds
            if counter == 0:
                fmt_degs = entry
            elif counter == 1:
                fmt_mins = entry
            elif counter == 2:
                fmt_sec = entry
            counter += 1
    if counter > 3:
        raise ValueError("too many format strings, only 3 numbers are allowed")

    result = []
    for degs, letters in zip([lat, lon], ["NS", "EW"]):
        # split sign
        neg = degs < 0
        degs = abs(degs)
        # get minutes
        mins = (degs * 60) % 60
        # get seconds
        secs = (mins * 60) % 60

        # if the seconds are rounded up to 60, increase mins
        if fmt_sec is not None and fmt_sec % secs == fmt_sec % 60:
            mins += 1
            secs = 0
        # if the mins are rounded up to 60 increase degs
        if fmt_mins is not None and fmt_mins % mins == fmt_mins % 60:
            degs += 1
            mins = 0

        # if no letter symbol is used, keep the sign
        if not use_letter and neg:
            degs = -degs

        # array of values and empty array of fills
        values = [degs, mins, secs]
        fills = []
        # gather the values to fill
        for entry in match:
            # the letter symbol
            if entry[-1] == "s":
                fills.append(letters[neg])
            # one of the values
            else:
                fills.append(values.pop(0))
        # format the string
        string = format % tuple(fills)
        # replace, if desired the degree sign
        if asLatex:
            string = string.replace("°", "\N{DEGREE SIGN}")
        # append to the results
        result.append(string)

    # return the results
    return result

def processDegree(data):
    # start with a value of 0
    value = 0
    # the degrees
    deg = data["deg"]
    # the minutes (optional)
    try:
        min = data["min"]
    except KeyError:
        min = 0
    # the seconds (optional)
    try:
        sec = data["sec"]
    except KeyError:
        sec = 0
    # the sign (optional)
    sign = data["sign"]
    if deg is not None:
        # convert the degrees absolute to float
        value += abs(float(deg))
        # but if there was a sign, store it
        if deg[0] == "-":
            sign = "S"
    # add the minutes
    if min is not None:
        value += float(min) / 60.
    # add the seconds
    if sec is not None:
        value += float(sec) / 3600.
    # add the sign
    if sign is not None:
        if sign in "SW":
            value *= -1
    # return the value
    return value

def gpsFromString(gps_string, height=None):
    """
    Read a gps coordinate from a text string in different formats, e.g. `70° 37’ 4.980" S 8° 9’ 26.280" W`,
    `70° 37.083 S 8° 9.438 W`, or `-70.618050° -8.157300°`.

    Parameters
    ----------
    gps_string : str, list
        the string of the point, containing both latitude and longitude, or a tuple with two strings one for latitude
        and one for longitude To batch process multiple strings, a list of strings can also be provided.
    height : float, optional
        the height of the gps point.

    Returns
    -------
    point : list
        a list containing, lat, lon, (height) of the given point.

    Examples
    --------

    >>> import cameratransform as ct

    Convert a coordinate string to a tuple:

    >>> ct.gpsFromString("85° 19′ 14″ N, 000° 02′ 43″ E")
    array([8.53205556e+01, 4.52777778e-02])

    Add a height information:

    >>> ct.gpsFromString("66° 39´56.12862´´S  140°01´20.39562´´ E", 13.769)
    array([-66.66559128, 140.02233212,  13.769     ])

    Use a tuple:

    >>> ct.gpsFromString(["66°39'56.12862''S", "140°01'20.39562'' E"])
    array([-66.66559128, 140.02233212])

    Or supply multiple coordinates with height information:

    >>> ct.gpsFromString([["-66.66559128° 140.02233212°", 13.769], ["66°39'58.73922''S  140°01'09.55709'' E", 13.769]])
    array([[-66.66559128, 140.02233212,  13.769     ],
           [-66.66631645, 140.01932141,  13.769     ]])
    """
    if not isinstance(gps_string, str):
        # keep a number
        if isinstance(gps_string, (float, int)):
            return gps_string
        # if it is a string and a number, interpret it as coordinates and height
        if len(gps_string) == 2 and isinstance(gps_string[0], str) and isinstance(gps_string[1], (float, int)):
            return gpsFromString(gps_string[0], gps_string[1])
        # recursively process it
        data = np.array([gpsFromString(data) for data in gps_string])
        # and optionally add a height
        if height is None:
            return data
        else:
            return np.hstack((data, [height]))
    regex_list = [r"(?P<deg>[\d+-]+)°\s*(?P<min>\d+)('|′|´|′)\s*(?P<sec>[\d.]+)(''|\"| |´´|″)\s*",
                  r"(?P<deg>[\d+-]+)°\s*(?P<min>[\d.]+)('|′|´|′)?\s*",
                  r"(?P<deg>[\d.+-]+)°\s*"]
    for string in regex_list:
        pattern = "\s*"+string.replace("<", "<lat_")+"(?P<lat_sign>N|S)?"+"\s*,?\s*"+string.replace("<", "<lon_")+"(?P<lon_sign>W|E)?"+"\s*"
        match = re.match(pattern, gps_string)
        if match:
            data = match.groupdict()
            gps = []
            for part in ["lat", "lon"]:
                value = processDegree({key[4:]:data[key] for key in data if key.startswith(part)})
                gps.append(value)
            if height is None:
                return np.array(gps)
            else:
                return np.array(gps + [height])
    # if not, try only a single coordinate
    for string in regex_list:
        pattern = "\s*"+string+"(?P<sign>N|S|W|E)?"+"\s*"
        match = re.match(pattern, gps_string)
        if match:
            data = match.groupdict()
            value = processDegree(data)
            return value


def getBearing(point1, point2):
    r"""
    The angle relative :math:`\beta` to the north direction from point :math:`(\mathrm{lat}_1, \mathrm{lon}_1)` to point :math:`(\mathrm{lat}_2, \mathrm{lon}_2)`:

    .. math::
        \Delta\mathrm{lon} &= \mathrm{lon}_2 - \mathrm{lon}_1\\
        X &= \cos(\mathrm{lat}_2) \cdot \sin(\Delta\mathrm{lon})\\
        Y &= \cos(\mathrm{lat}_1) \cdot \sin(\mathrm{lat}_2) - \sin(\mathrm{lat}_1) \cdot \cos(\mathrm{lat}_2) \cdot \cos(\Delta\mathrm{lon})\\
        \beta &= \arctan2(X, Y)

    Parameters
    ----------
    point1 : ndarray
        the first point from which to calculate the bearing, dimensions (2), (3), (Nx2), (Nx3)
    point2 : ndarray
        the second point to which to calculate the bearing, dimensions (2), (3), (Nx2), (Nx3)

    Returns
    -------
    bearing : float, ndarray
        the bearing angle in degree, dimensions (), (N)

    Examples
    --------

    >>> import cameratransform as ct

    Calculate the bearing in degrees between two gps positions:

    >>> ct.getBearing([85.3205556, 4.52777778], [-66.66559128, 140.02233212])
    53.34214977328738

    or between a list of gps positions:

    >>> ct.getBearing([[85.3205556, 4.52777778], [65.3205556, 7.52777778]], [[-66.66559128, 140.02233212], [-60.66559128, 80.02233212]])
    array([ 53.34214977, 136.82109976])

    """
    lat1, lon1, h1 = splitGPS(point1)
    lat2, lon2, h2 = splitGPS(point2)
    dL = lon2-lon1
    X = np.cos(lat2) * np.sin(dL)
    Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dL)
    beta = np.arctan2(X, Y)
    return np.rad2deg(beta)

def splitGPS(x, keep_deg=False):
    x = np.array(x)
    if keep_deg is False:
        lat1 = np.deg2rad(x[..., 0])
        lon1 = np.deg2rad(x[..., 1])
    else:
        lat1 = x[..., 0]
        lon1 = x[..., 1]
    try:
        h1 = x[..., 2]
    except IndexError:
        h1 = None
    return lat1, lon1, h1

def getDistance(point1, point2):
    r"""
    Calculate the great circle distance between two points :math:`(\mathrm{lat}_1, \mathrm{lon}_1)` and :math:`(\mathrm{lat}_2, \mathrm{lon}_2)`
    on the earth (specified in decimal degrees)

    .. math::
        \Delta\mathrm{lon} &= \mathrm{lon}_2 - \mathrm{lon}_1\\
        \Delta\mathrm{lat} &= \mathrm{lat}_2 - \mathrm{lat}_1\\
        a &= \sin(\Delta\mathrm{lat}/2)^2 + \cos(\mathrm{lat}_1) \cdot \cos(\mathrm{lat}_2) \cdot \sin(\Delta\mathrm{lat}/2)^2\\
        d &= 6371\,\mathrm{km} \cdot 2 \arccos(\sqrt a)

    Parameters
    ----------
    point1 : ndarray
        the start point from which to calculate the distance, dimensions (2), (3), (Nx2), (Nx3)
    point2 : ndarray
        the end point to which to calculate the distance, dimensions (2), (3), (Nx2), (Nx3)

    Returns
    -------
    distance : float, ndarray
        the distance in m, dimensions (), (N)

    Examples
    --------

    >>> import cameratransform as ct

    Calculate the distance in m between two gps positions:

    >>> ct.getDistance([52.51666667, 13.4], [48.13583333, 11.57988889])
    503926.75849507266

    or between a list of gps positions:

    >>> ct.getDistance([[52.51666667, 13.4], [52.51666667, 13.4]], [[49.597854, 11.005092], [48.13583333, 11.57988889]])
    array([365127.04999716, 503926.75849507])

    """
    lat1, lon1, h1 = splitGPS(point1)
    lat2, lon2, h2 = splitGPS(point2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371e3 * c

    if h1 is not None and h2 is not None:
        dH = np.abs(h1 - h2)
        distance = np.sqrt(distance**2 + dH**2)

    return distance

def moveDistance(start, distance, bearing):
    r"""
    Moving from :math:`(\mathrm{lat}_1, \mathrm{lon}_1)` a distance of :math:`d` in the direction of :math:`\beta`:

    .. math::
        R &= 6371\,\mathrm{km}\\
        \mathrm{lat}_2 &= \arcsin(\sin(\mathrm{lat}_1) \cdot \cos(d / R) +
                         \cos(\mathrm{lat}_1) \cdot \sin(d / R) \cdot \cos(\beta))\\
        \mathrm{lon}_2 &= \mathrm{lon}_1 + \arctan\left(\frac{\sin(\beta) \cdot \sin(d / R) \cdot \cos(\mathrm{lat}_1)}{
                                 \cos(d / R) - \sin(\mathrm{lat}_1) \cdot \sin(\mathrm{lat}_2)}\right)

    Parameters
    ----------
    start : ndarray
        the start point from which to calculate the distance, dimensions (2), (3), (Nx2), (Nx3)
    distance : float, ndarray
        the distance to move in m, dimensions (), (N)
    bearing : float, ndarray
        the bearing angle in degrees, specifying in which direction to move, dimensions (), (N)

    Returns
    -------
    target : ndarray
        the target point, dimensions (2), (3), (Nx2), (Nx3)

    Examples
    --------

    >>> import cameratransform as ct

    Move from 52.51666667°N 13.4°E, 503.926 km in the direction -164°:

    >>> ct.moveDistance([52.51666667, 13.4], 503926, -164)
    array([48.14444416, 11.52952357])

    Batch process multiple positions at once:

    >>> ct.moveDistance([[52.51666667, 13.4], [49.597854, 11.005092]], [10, 20], -164)
    array([[52.51658022, 13.39995926],
           [49.5976811 , 11.00501551]])

    Or one positions in multiple ways:

    >>> ct.moveDistance([52.51666667, 13.4], [503926, 103926], [-164, -140])
    array([[48.14444416, 11.52952357],
           [51.79667095, 12.42859387]])
    """
    start = np.array(start)
    distance = np.array(distance)
    bearing = np.deg2rad(bearing)
    lat1, lon1, h1 = splitGPS(start)
    R = 6371e3
    if start.shape[-1] == 3:
        R += start[..., 2]
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance / R) +
                     np.cos(lat1) * np.sin(distance / R) * np.cos(bearing))

    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance / R) * np.cos(lat1),
                             np.cos(distance / R) - np.sin(lat1) * np.sin(lat2))
    if start.shape[-1] == 3:
        return np.array([np.rad2deg(lat2), np.rad2deg(lon2), np.ones_like(lon2)*start[..., 2]]).T
    return np.array([np.rad2deg(lat2), np.rad2deg(lon2)]).T


def spaceFromGPS(gps, gps0):
    if len(gps[..., :]) == 2:
        height = np.zeros_like(gps[..., 0])
    else:
        height = gps[..., 2]
    distance = getDistance(gps0, gps)
    bearing_rad = np.deg2rad(getBearing(gps0, gps))
    return np.array([distance * np.sin(bearing_rad), distance * np.cos(bearing_rad), height]).T


def gpsFromSpace(space, gps0):
    bearing = np.rad2deg(np.arctan2(space[..., 0], space[..., 1]))
    distance = np.linalg.norm(space[..., :2], axis=-1)
    target = moveDistance(gps0, distance, bearing)
    if space.shape[-1] == 3:
        return np.array([target[..., 0], target[..., 1], space[..., 2]]).T
    return target
