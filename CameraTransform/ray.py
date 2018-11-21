import numpy as np


def my_inner(a, b):
    return np.einsum('...k,...k->...', a, b)


def ray_intersect_triangle(origin, direction, triangle, use_planes=False):
    """
    This function can intersect R rays with T triangles and return the intersection points.
    source: http://geomalgorithms.com/a06-_intersect-2.html

    Parameters
    ----------
    origin : ndarray
        the origin point(s) of the ray(s), dimensions: (3) or (R,3)
    direction : ndarray
        the direction vector(s) of the ray(s), dimensions: (3) or (R,3)
    triangle : ndarray
        the triangle(s) to intersect the ray(s), dimensions: (3,3), or (T,3,3)
    use_planes : bool
        whether to allow intersections outside the triangle (or whether to interpret the triangle as a plane).

    Returns
    -------
    points : ndarray
        the intersection point(s) of the ray(s) with the triangle(s), dimensions: (3) or (R,3). Points have nan values
        when there is no intersection.
    """
    origin = np.array(origin)
    direction = np.array(direction)
    if len(direction.shape) == 1:
        direction = direction.reshape(1, *direction.shape)
        return_single = True
    else:
        return_single = False
    triangle = np.array(triangle)
    if len(triangle.shape) == 2:
        triangle = triangle.reshape(1, *triangle.shape)

    v0 = triangle[..., 0, :]
    v1 = triangle[..., 1, :]
    v2 = triangle[..., 2, :]
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    b = np.inner(normal, direction)
    a = my_inner(normal[..., None, :], v0[..., None, :] - origin[None, ..., :])

    rI = a / b
    # ray is parallel to the plane
    rI[(b == 0.0)*(a != 0.0)] = np.nan
    # ray is parallel and lies in the plane
    rI[(b == 0.0)*(a == 0.0)] = 0

    # check whether the intersection is behind the origin of the ray
    rI[rI < 0.0] = np.nan

    if not use_planes:
        w = origin + rI[..., None] * direction - v0[..., None, :]
        denom = my_inner(u, v) * my_inner(u, v) - my_inner(u, u) * my_inner(v, v)

        si = (my_inner(u, v)[..., None] * my_inner(w, v[..., None, :]) - my_inner(v, v)[..., None] * my_inner(w, u[..., None, :])) / denom[:, None]
        rI[((si < 0)+(si > 1.0)).astype(bool)] = np.nan

        ti = (my_inner(u, v)[..., None] * my_inner(w, u[..., None, :]) - my_inner(u, u)[..., None] * my_inner(w, v[..., None, :])) / denom[:, None]
        rI[((ti < 0.0) + (si + ti > 1.0)).astype(bool)] = np.nan

    def nanargmin(a, axis):
        from numpy.lib.nanfunctions import _replace_nan
        a, mask = _replace_nan(a, np.inf)
        res = np.argmin(a, axis=axis)
        return res

    index = nanargmin(rI, axis=0)
    rI = rI[index, np.arange(len(index))]
    point = origin + rI[..., None] * direction

    if return_single:
        return point[0]
    return point


def intersectionOfTwoLines(p1, v1, p2, v2):
    """
    Get the point closest to the intersection of two lines. The lines are given by one point (p1 and p2) and a
    direction vector v (v1 and v2). If a batch should be processed, the multiple direction vectors can be given.
    """
    # if we transform multiple points in one go
    if len(v1.shape) == 2:
        a1 = np.einsum('ij,ij->i', v1, v1)
        a2 = np.einsum('ij,ij->i', v1, v2)
        b1 = -np.einsum('ij,ij->i', v2, v1)
        b2 = -np.einsum('ij,ij->i', v2, v2)
        c1 = -np.einsum('ij,j->i', v1, p1 - p2)
        c2 = -np.einsum('ij,j->i', v2, p1 - p2)
        res = np.linalg.solve(np.array([[a1, b1], [a2, b2]]).transpose(2, 0, 1), np.array([c1, c2]).T)
        res = res[:, None, :]
        return np.mean([p1 + res[..., 0] * v1, p2 + res[..., 1] * v2], axis=0)
    else:  # or just one point
        a1 = np.dot(v1, v1)
        a2 = np.dot(v1, v2)
        b1 = -np.dot(v2, v1)
        b2 = -np.dot(v2, v2)
        c1 = -np.dot(v1, p1 - p2)
        c2 = -np.dot(v2, p1 - p2)
        res = np.linalg.solve(np.array([[a1, b1], [a2, b2]]), np.array([c1, c2]))
        res = res[None, None, :]
        return np.mean([p1 + res[..., 0] * v1, p2 + res[..., 1] * v2], axis=0)[0]


def distanceOfTwoLines(p1, v1, p2, v2):
    """
    Get the point closest to the intersection of two lines. The lines are given by one point (p1 and p2) and a
    direction vector v (v1 and v2). If a batch should be processed, the multiple direction vectors can be given.
    """
    # if we transform multiple points in one go
    if len(v1.shape) == 2:
        a1 = np.einsum('ij,ij->i', v1, v1)
        a2 = np.einsum('ij,ij->i', v1, v2)
        b1 = -np.einsum('ij,ij->i', v2, v1)
        b2 = -np.einsum('ij,ij->i', v2, v2)
        c1 = -np.einsum('ij,j->i', v1, p1 - p2)
        c2 = -np.einsum('ij,j->i', v2, p1 - p2)
        res = np.linalg.solve(np.array([[a1, b1], [a2, b2]]).transpose(2, 0, 1), np.array([c1, c2]).T)
        res = res[:, None, :]
        #print("res[..., 0]", res[..., 0], res[..., 1], np.linalg.norm((p1 + res[..., 0] * v1) - (p2 + res[..., 1] * v2), axis=1))
        return np.linalg.norm((p1 + res[..., 0] * v1) - (p2 + res[..., 1] * v2), axis=1)#+(res[..., 0]<10)*999+(res[..., 1]<10)*999
    else:  # or just one point
        a1 = np.dot(v1, v1)
        a2 = np.dot(v1, v2)
        b1 = -np.dot(v2, v1)
        b2 = -np.dot(v2, v2)
        c1 = -np.dot(v1, p1 - p2)
        c2 = -np.dot(v2, p1 - p2)
        res = np.linalg.solve(np.array([[a1, b1], [a2, b2]]), np.array([c1, c2]))
        res = res[None, None, :]
        return np.linalg.norm((p1 + res[..., 0] * v1) - (p2 + res[..., 1] * v2), axis=1)[0]

def extrudeLine(points, z0, z1):
    mesh = []
    last_point = None
    for point in points:
        point = list(point)
        if last_point is not None:
            mesh.append([point + [z0], last_point + [z0], point + [z1]])
            mesh.append([point + [z0], last_point + [z0], last_point + [z1]])
        last_point = point
    return np.array(mesh)
