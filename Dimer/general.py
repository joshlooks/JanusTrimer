import numpy as np
from scipy.integrate import nquad as nint
import time
from numba import njit
import concurrent.futures


##-----------------------------------------------------------------------------------------
## Important Functions

# Coordinate Transform
@njit
def cart(theta1, phi1, rp, thetap, phip):  # converts rotated sphericals (rp,thetap,phip) to Cartesians (x,y,z), given the rotation theta1, phi1
    st1 = np.sin(theta1)
    ast1 = np.abs(st1)
    ct1 = np.cos(theta1)
    sp1 = np.sin(phi1)
    cp1 = np.cos(phi1)
    stp = np.sin(thetap)
    ctp = np.cos(thetap)
    spp = np.sin(phip)
    cpp = np.sin(phip)
    
    return np.array([rp*stp*cpp*st1*sp1/ast1 + rp*stp*spp*st1*ct1*cp1/ast1 + rp*ctp*st1*cp1, -rp*stp*cpp*st1*cp1/ast1 + rp*stp*spp*st1*ct1*sp1/ast1 + rp*ctp*st1*sp1, -rp*stp*spp*ast1 + rp*ctp*ct1])

# Distances
@njit
def rhomo(rp, thetap, phip, rpp, thetapp, phipp):  # dist between two points in the same spherical coordinates
    pos1p = np.array([rp*np.sin(thetap)*np.cos(phip), rp*np.sin(thetap)*np.sin(phip), rp*np.cos(thetap)])
    pos2p = np.array([rpp*np.sin(thetapp)*np.cos(phipp), rpp*np.sin(thetapp)*np.sin(phipp), rpp*np.cos(thetapp)])
    d = pos1p-pos2p
    return np.sqrt(np.dot(d,d))

@njit
def rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, rpp, thetapp, phipp, x, y, z):  # dist between two points in different spherical coordinates, where one sphere is at the origin and another is at (x,y,z)
    pos1 = cart(theta1, phi1, rp, thetap, phip)
    pos2 = cart(theta2, phi2, rpp, thetapp, phipp) + np.array([x,y,z])
    d = pos1-pos2
    return np.sqrt(np.dot(d,d))

# Lennard-Jones Potential Without Multiplying Constants
@njit
def rljp(sigma, r):
    x = (sigma/r)**6
    return x**2-x


##-----------------------------------------------------------------------------------------
## Simulation Parameters

# LJ units
epsilon = 1.0  
sigma = 1.0

# LJ epsilon coefficients for different interactions
aa = 1.0  # Hydrophillic-Hydrophillic
ab = 0.05  # Hydrophillic-Hydrophobic
bb = 0.0025  # Hydrophobic-Hydrophobic
aw = 1.0  # Hydrophillic-Fluid
bw = 0.05  # Hydrophobic-Fluid

# Distances
R = 6.0*sigma  # Sphere radius
d = 0.51*sigma  # atomic spacing

# Atom density
surfaceeta = 1.3/sigma**2
volumeeta = 0.77/sigma**3

# Sphere Orientations

theta1 = (180-0.1)/180*np.pi

phi1 = 0.0

theta2 = 0.1/180*np.pi

phi2 = 0.0/180*np.pi 


##-----------------------------------------------------------------------------------------
## Integrands

@njit
def integrand1(phipp, thetapp, phip, thetap):
    r = rhetero(theta1, phi1, R, thetap, phip, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    Vaa = rljp(sigma, r)
    return Vaa*R**4*np.sin(thetap)*np.sin(thetapp)

@njit
def integrand2(phipp, thetapp, phip, thetap):
    r = rhetero(theta1, phi1, R, thetap, phip, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    Vab = rljp(sigma, r)
    return Vab*R**4*np.sin(thetap)*np.sin(thetapp)

@njit
def integrand3(phipp, thetapp, phip, thetap):
    r = rhetero(theta1, phi1, R, thetap, phip, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    Vab = rljp(sigma, r)
    return Vab*R**4*np.sin(thetap)*np.sin(thetapp)

@njit
def integrand4(phipp, thetapp, phip, thetap):
    r = rhetero(theta1, phi1, R, thetap, phip, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    Vbb = rljp(sigma, r)
    return Vbb*R**4*np.sin(thetap)*np.sin(thetapp)

@njit
def integrand5_A(phip_1, thetap_1, rp_1, phip_2, thetap_2):
    r = rhomo(rp_1, thetap_1, phip_1, R, thetap_2, phip_2)
    Vaw = rljp(sigma, r)
    return Vaw*rp_1**2*np.sin(thetap_1)*R**2*np.sin(thetap_2)

@njit
def integrand5_B(phipp, thetapp, rpp, phip, thetap):
    r = rhetero(theta2, phi2, rpp, thetapp, phipp, theta1, phi1, R, thetap, phip, 0.0, 0.0, (2*R+d))
    Vaw = rljp(sigma, r)
    return Vaw*rpp**2*np.sin(thetapp)*R**2*np.sin(thetap)

@njit
def integrand6_A(phip_1, thetap_1, rp_1, phip_2, thetap_2):
    r = rhomo(rp_1, thetap_1, phip_1, R, thetap_2, phip_2)
    Vbw = rljp(sigma, r)
    return Vbw*rp_1**2*np.sin(thetap_1)*R**2*np.sin(thetap_2)

@njit
def integrand6_B(phipp, thetapp, rpp, phip, thetap):
    r = rhetero(theta2, phi2, rpp, thetapp, phipp, theta1, phi1, R, thetap, phip, 0.0, 0.0, (2*R+d))
    Vbw = rljp(sigma, r)
    return Vbw*rpp**2*np.sin(thetapp)*R**2*np.sin(thetap)

@njit
def integrand7_A(phipp_1, thetapp_1, rpp_1, phipp_2, thetapp_2):
    r = rhomo(rpp_1, thetapp_1, phipp_1, R, thetapp_2, phipp_2)
    Vaw = rljp(sigma, r)
    return Vaw*rpp_1**2*np.sin(thetapp_1)*R**2*np.sin(thetapp_2)

@njit
def integrand7_B(phip, thetap, rp, phipp, thetapp):
    r = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    Vaw = rljp(sigma, r)
    return Vaw*rp**2*np.sin(thetap)*R**2*np.sin(thetapp)

@njit
def integrand8_A(phipp_1, thetapp_1, rpp_1, phipp_2, thetapp_2):
    r = rhomo(rpp_1, thetapp_1, phipp_1, R, thetapp_2, phipp_2)
    Vbw = rljp(sigma, r)
    return Vbw*rpp_1**2*np.sin(thetapp_1)*R**2*np.sin(thetapp_2)

@njit
def integrand8_B(phip, thetap, rp, phipp, thetapp):
    r = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    Vbw = rljp(sigma, r)
    return Vbw*rp**2*np.sin(thetap)*R**2*np.sin(thetapp)


##-----------------------------------------------------------------------------------------
## Energy Integrals defined as functions to be called in the multiprocessing pool executor

def e1():
    e = nint(integrand1, [[0.0, 2*np.pi],[np.pi/2, np.pi],[0.0, 2*np.pi],[np.pi/2, np.pi]])
    return surfaceeta**2*4*epsilon*aa*e[0]


def e2():
    e = nint(integrand2, [[0.0, 2*np.pi],[0.0, np.pi/2],[0.0, 2*np.pi],[np.pi/2, np.pi]])
    return surfaceeta**2*4*epsilon*ab*e[0]


def e3():
    e = nint(integrand3, [[0.0, 2*np.pi],[np.pi/2, np.pi],[0.0, 2*np.pi],[0.0, np.pi/2]])
    return surfaceeta**2*4*epsilon*ab*e[0]


def e4():
    e = nint(integrand4, [[0.0, 2*np.pi],[0.0, np.pi/2],[0.0, 2*np.pi],[0.0, np.pi/2]])
    return surfaceeta**2*4*epsilon*bb*e[0]


def e5_A():
    return 25648.459899401107


def e5_B():
    e = nint(integrand5_B, [[0.0, 2*np.pi],[0.0, np.pi],[0.0, R],[0.0, 2*np.pi],[np.pi/2, np.pi]])
    return -surfaceeta*volumeeta*4*epsilon*aw*e[0]


def e6_A():
    return 1282.42299497006


def e6_B():
    e = nint(integrand6_B, [[0.0, 2*np.pi],[0.0, np.pi],[0.0, R],[0.0, 2*np.pi],[0.0, np.pi/2]])
    return -surfaceeta*volumeeta*4*epsilon*bw*e[0]


def e7_A():
    return 25648.459899401107


def e7_B():
    e = nint(integrand7_B, [[0.0, 2*np.pi],[0.0, np.pi],[0.0, R],[0.0, 2*np.pi],[np.pi/2, np.pi]])
    return -surfaceeta*volumeeta*4*epsilon*aw*e[0]


def e8_A():
    return 1282.42299497006


def e8_B():
    e = nint(integrand8_B, [[0.0, 2*np.pi],[0.0, np.pi],[0.0, R],[0.0, 2*np.pi],[0.0, np.pi/2]])
    return -surfaceeta*volumeeta*4*epsilon*bw*e[0]


##-----------------------------------------------------------------------------------------
## Integrating unsing multiprocessing
## Submits each integral to a logical core as it becomes available
## Note that all subsequent code must be kept within the if __name__ == '__main__': statement

Energy = 0.0    # Initialising energy variable

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:

        f1 = executor.submit(e1,)
        f2 = executor.submit(e2,)
        f3 = executor.submit(e3,)
        f4 = executor.submit(e4,)
        f5_A= executor.submit(e5_A,)
        f5_B = executor.submit(e5_B,)
        f6_A = executor.submit(e6_A,)
        f6_B = executor.submit(e6_B,)
        f7_A = executor.submit(e7_A,)
        f7_B = executor.submit(e7_B,)
        f8_A = executor.submit(e8_A,)
        f8_B = executor.submit(e8_B,)

        for f in concurrent.futures.as_completed([f1,f2,f3,f4,f5_A,f5_B,f6_A,f6_B,f7_A,f7_B,f8_A,f8_B]):
            Energy += f.result()


    
    print(Energy)
