import numpy as np
import matplotlib.pyplot as plt

def gaussian(my, mx, sigma, y, x):
    return np.exp(-0.5*((y-my)**2+(x-mx)**2)/sigma**2)

def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

def lattice(unitcell, lim):
    """
    Generate a lattice of points based on a unit cell and some limited window
    inputs
    ======
    unitcell: array_like (2, 2) whose columns represent the unit cell
    lim: array_like (2) defining the limits on a rectangular window

    returns:
        lattice points: array_like (N, 2) where N is determined by how many
        unitcell points fit inside the window
    """
    v0, v1 = unitcell.T
    corners = np.array([[0., 0.], [0., lim[1]], [lim[0], 0], lim])
    # find corners of image in unitcell basis
    newcorners = [np.linalg.solve(unitcell, c) for c in corners] 
    min0, min1 = np.floor(np.min(newcorners, 0)).astype('int')
    max0, max1 = np.ceil(np.max(newcorners, 0)).astype('int')
    pts = []
    for i in range(min0, max0+1):
        for j in range(min1, max1+1):
            pts.append(i*v0 + j*v1)
    return np.array(pts)

def latticeimg(lattice, sigma, y, x):
    img = np.zeros_like(y*x)
    for l in lattice:
        img += gaussian(l[0], l[1], sigma, y, x)
    return img

# making a square lattice with a unit cell consisting of two atoms of different
# intensities

def twositesquarelattice(shifts, L=64, angle=-np.pi/5., a=1./6, sigma=1./25, 
                         bsiteintensity=0.7, noise=0., seed=14850):
    ymin, xmin = np.min(np.concatenate([shifts, np.zeros((1,2))]), 0)
    ymax, xmax = np.max(np.concatenate([shifts, np.ones((1,2))]), 0)
    y = np.linspace(ymin, ymax, L)[:, None]
    x = np.linspace(xmin, xmax, L)[None, :]
    
    unitcell = a * rot(-np.pi/4.5)  # slightly incommensurate with the window
    offset = unitcell.sum(1) / 2.
    
    sites = lattice(unitcell, np.ones(2))
    rng = np.random.RandomState(seed)
    asites = sites + noise * rng.randn(len(sites), 2)
    bsites = sites - offset[None,:] + noise * rng.randn(len(sites), 2)
    
    model = []
    for s in shifts:
        ashiftsites = asites - s
        bshiftsites = bsites - s
        img = latticeimg(ashiftsites, sigma, y, x)
        img += bsiteintensity * latticeimg(bshiftsites, sigma, y, x)
        model.append(img)
    model = np.array(model)
    return model/model.max()

def twoparticles(shifts, L=64, pos=[0.4*np.random.rand(2)+0.05,
                                    0.4*np.random.rand(2)+0.5],
                 widths=[0.2, 0.17], amps=[.8, 1.]):
    y = np.linspace(0, 1, L)[:, None]
    x = np.linspace(0, 1, L)[None, :]
    
    model = []
    for s in shifts:
        p0, p1 = pos - s[None, :]
        img = amps[0] * gaussian(p0[0], p0[1], widths[0], y, x)
        img += amps[1] * gaussian(p1[0], p1[1], widths[1], y, x)
        model.append(img)
    model = np.array(model)
    return model/model.max()
