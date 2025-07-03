import numpy as np

def string_to_shape(shape_string):
    parts = shape_string.split("x")
    shape = []
    for part in parts:
        try:
            part = int(part)
            shape.append(part)
        except:
            shape.append(part)
    return shape

def ensure_array_format(p, shape, *additional_arrays):
    if len(additional_arrays) > 0:
        additional_arrays = list(additional_arrays)
        additional_arrays.insert(0, p)
        additional_arrays = [np.asarray(a) for a in additional_arrays]
        assert np.all([a.shape == additional_arrays[0].shape for a in additional_arrays]), "All arrays must have the same shape"
        return [ensure_array_format(a, shape) for a in additional_arrays]
    if isinstance(shape, str):
        shape = string_to_shape(shape)
    p = np.asarray(p)
    number_lookup = {}
    if shape[0] != "..." and len(p.shape) != len(shape):
        raise ValueError("Expected array of shape %s, got array of shape %s" % (str(shape), str(p.shape)))
    for i in range(-1, -len(shape)-1, -1):
        if isinstance(shape[i], str):
            if shape[i] == "...":
                break
            if shape[i] in number_lookup:
                assert number_lookup[shape[i]] == p.shape[i], "Expected array of shape %s, got array of shape %s" % (str(shape), str(p.shape))
            number_lookup[shape[i]] = p.shape[i]
        else:
            assert p.shape[i] == shape[i], "Expected array of shape %s, got array of shape %s" % (str(shape), str(p.shape))
    return p


if __name__ == '__main__':
    ensure_array_format(np.random.rand(3), "3")
    print("---")
    ensure_array_format(np.random.rand(3), "...x3")
    ensure_array_format(np.random.rand(5, 4), "Nx4")
