from __future__ import print_function

def get_canonical_kcenter(dims):
    """

    get the galsim canonical center for an image in k space.  See
    the drawKImage method

    """
    assert (dims[0] % 2)==0
    assert (dims[1] % 2)==0

    cen=[
        int( (dims[0]-1.0)/2.0 + 0.5),
        int( (dims[1]-1.0)/2.0 + 0.5),
    ]

    return cen


