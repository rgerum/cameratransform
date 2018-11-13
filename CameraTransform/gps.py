import numpy as np
import re


def formatGPS(lat, lon, format=None, asLatex=False):
    """ Formats a latiture, longitude pair in degress according to the format string.
        The format string can contain a %s, to denote the letter symbol (N, S, W, E) and up to three number formaters (%d or %f), to
        denote the degrees, minutes and seconds. To not lose precision, the last one can be float number.

        common formats are e.g.:

           +--------------------------------+--------------------------------------+
           | format                         | output                               |
           +================================+===================+==================+
           | %2d° %2d' %6.3f" %s (default)  | 70° 37'  4.980" S | 8°  9' 26.280" W |
           +--------------------------------+-------------------+------------------+
           | %2d° %2d.3f %s                 | 70° 37.083 S      | 8°  9.438 W      |
           +--------------------------------+-------------------+------------------+
           | %2d°                           | -70.618050°       | -8.157300°       |
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

        >>> import CameraTransform as ct

        Convert a coordinate pair to a formatted string:

        >>> lat, lon = ct.formatGPS(-70.61805, -8.1573)
        >>> print(lat)
        70° 37'  4.980" S
        >>> print(lon)
         8°  9' 26.280" W

        or with a different format:

        >>> lat, lon = ct.formatGPS(-70.61805, -8.1573, format="%2d° %2d.3f %s")
        >>> print(lat)
        70° 37.3f S
        >>> print(lon)
         8°  9.3f W

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

def gpsFromString(gps_string, height=None):
    regex_list = [r"(?P<deg>[\d+-]+)°\s*(?P<min>\d+)'\s*(?P<sec>[\d.]+)(''|\"| )\s*",
                  r"(?P<deg>[\d+-]+)°\s*(?P<min>[\d.]+)'\s*",
                  r"(?P<deg>[\d.+-]+)°\s*"]
    for string in regex_list:
        match = re.match(string.replace("<", "<lat_")+"(?P<lat_sign>N|S)?"+"\s*"+string.replace("<", "<lon_")+"(?P<lon_sign>W|E)?", gps_string)
        if match:
            data = match.groupdict()
            gps = []
            for part in ["lat", "lon"]:
                value = 0
                deg = data[part+"_deg"]
                min = data[part+"_min"]
                sec = data[part+"_sec"]
                sign = data[part+"_sign"]
                if deg is not None:
                    value += float(deg)
                if min is not None:
                    value += float(min)/60
                if sec is not None:
                    value += float(sec)/3600
                if sign is not None:
                    if sign in "SW":
                        value *= -1
                gps.append(value)
            if height is None:
                return gps
            else:
                return gps + [height]


def getBearing(x, y):
    lat1, lon1, h1 = splitGPS(x)
    lat2, lon2, h2 = splitGPS(y)
    dL = lon2-lon1
    X = np.cos(lat2) * np.sin(dL)
    Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dL)
    beta = np.arctan2(X, Y)
    return np.rad2deg(beta)

def splitGPS(x):
    x = np.array(x)
    lat1 = np.deg2rad(x[..., 0])
    lon1 = np.deg2rad(x[..., 1])
    try:
        h1 = x[..., 2]
    except IndexError:
        h1 = None
    return lat1, lon1, h1

def getDistance(x, y):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lat1, lon1, h1 = splitGPS(x)
    lat2, lon2, h2 = splitGPS(y)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371e3 * c

    if h1 is not None:
        dH = np.abs(h1 - h2)
        km = np.sqrt(km**2 + dH**2)

    return km

def spaceFromGPS(gps, gps0):
    distance = getDistance(gps0, gps)
    bearing = getBearing(gps0, gps)
    return np.array([distance * np.sin(np.deg2rad(bearing)), distance * np.cos(np.deg2rad(bearing)), gps[..., 2]]).T

def gpsFromSpace(space, gps0):
    lat1, lon1, h1 = splitGPS(gps0)
    bearing = np.arctan2(space[..., 1], space[..., 0])
    distance = np.linalg.norm(space, axis=-1)
    R = 6371e3
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance / R) +
                     np.cos(lat1) * np.sin(distance / R) * np.cos(bearing))

    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance / R) * np.cos(lat1),
                             np.cos(distance / R) - np.sin(lat1) * np.sin(lat2))
    return np.array([np.rad2deg(lat2), np.rad2deg(lon2), space[..., 2]]).T
