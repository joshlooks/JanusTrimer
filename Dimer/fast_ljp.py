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
    cpp = np.cos(phip)
    
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

A = 4*epsilon*sigma**6
B = 4*epsilon*sigma**12

# Distances
R = 6.0*sigma  # Sphere radius
d = 0.51*sigma  # atomic spacing

# Atom density
surfaceeta = 1.3/sigma**2
volumeeta = 0.77/sigma**3

# Sphere Orientations

theta1 = 50.01/180*np.pi

phi1 = 0.0

theta2 = 130.01/180*np.pi

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
def integrand5_B(s, phip, thetap):
    delta = rhetero(theta2, phi2, 0.0, 0.0, 0.0, theta1, phi1, R, thetap, phip, 0.0, 0.0, (2*R+d))
    return R**2*np.sin(thetap)*s/delta*(-A/2*(1/(delta-s)**4-1/(delta+s)**4)+B/5*(1/(delta-s)**10-1/(delta+s)**10))

@njit
def integrand6_A(phip_1, thetap_1, rp_1, phip_2, thetap_2):
    r = rhomo(rp_1, thetap_1, phip_1, R, thetap_2, phip_2)
    Vbw = rljp(sigma, r)
    return Vbw*rp_1**2*np.sin(thetap_1)*R**2*np.sin(thetap_2)

@njit
def integrand6_B(s, phip, thetap):
    delta = rhetero(theta2, phi2, 0.0, 0.0, 0.0, theta1, phi1, R, thetap, phip, 0.0, 0.0, (2*R+d))
    return R**2*np.sin(thetap)*s/delta*(-A/2*(1/(delta-s)**4-1/(delta+s)**4)+B/5*(1/(delta-s)**10-1/(delta+s)**10))

@njit
def integrand7_A(phipp_1, thetapp_1, rpp_1, phipp_2, thetapp_2):
    r = rhomo(rpp_1, thetapp_1, phipp_1, R, thetapp_2, phipp_2)
    Vaw = rljp(sigma, r)
    return Vaw*rpp_1**2*np.sin(thetapp_1)*R**2*np.sin(thetapp_2)

@njit
def integrand7_B(s, phipp, thetapp):
    delta = rhetero(theta1, phi1, 0.0, 0.0, 0.0, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    return R**2*np.sin(thetapp)*s/delta*(-A/2*(1/(delta-s)**4-1/(delta+s)**4)+B/5*(1/(delta-s)**10-1/(delta+s)**10))

@njit
def integrand8_A(phipp_1, thetapp_1, rpp_1, phipp_2, thetapp_2):
    r = rhomo(rpp_1, thetapp_1, phipp_1, R, thetapp_2, phipp_2)
    Vbw = rljp(sigma, r)
    return Vbw*rpp_1**2*np.sin(thetapp_1)*R**2*np.sin(thetapp_2)

@njit
def integrand8_B(s, phipp, thetapp):
    delta = rhetero(theta1, phi1, 0.0, 0.0, 0.0, theta2, phi2, R, thetapp, phipp, 0.0, 0.0, -(2*R+d))
    return R**2*np.sin(thetapp)*s/delta*(-A/2*(1/(delta-s)**4-1/(delta+s)**4)+B/5*(1/(delta-s)**10-1/(delta+s)**10))


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
    e = nint(integrand5_B, [[0.0, R],[0.0, 2*np.pi],[np.pi/2, np.pi]])
    return -np.pi*surfaceeta*volumeeta*aw*e[0]


def e6_A():
    return 1282.42299497006


def e6_B():
    e = nint(integrand6_B, [[0.0, R],[0.0, 2*np.pi],[0.0, np.pi/2]])
    return -np.pi*surfaceeta*volumeeta*bw*e[0]


def e7_A():
    return 25648.459899401107


def e7_B():
    e = nint(integrand7_B, [[0.0, R],[0.0, 2*np.pi],[np.pi/2, np.pi]])
    return -np.pi*surfaceeta*volumeeta*aw*e[0]


def e8_A():
    return 1282.42299497006


def e8_B():
    e = nint(integrand8_B, [[0.0, R],[0.0, 2*np.pi],[0.0, np.pi/2]])
    return -np.pi*surfaceeta*volumeeta*bw*e[0]


##-----------------------------------------------------------------------------------------
## Energy Integral

tstart = time.time()

e1 = e1()
t1tot = time.time()
t1 = t1tot - tstart
print("1 =", e1, "In", t1/60, "minutes (", t1/3600, "hours )")

e2 = e2()
t2tot = time.time()
t2 = t2tot - t1tot
print("2 =", e2, "In", t2/60, "minutes (", t2/3600, "hours )")

e3 = e3()
t3tot = time.time()
t3 = t3tot - t2tot
print("3 =", e3, "In", t3/60, "minutes (", t3/3600, "hours )")

e4 = e4()
t4tot = time.time()
t4 = t4tot - t3tot
print("4 =", e4, "In", t4/60, "minutes (", t4/3600, "hours )")

e5_A = e5_A()
t5Atot = time.time()
t5A = t5Atot - t4tot
print("5_A =", e5_A, "In", t5A/60, "minutes (", t5A/3600, "hours )")

e5_B = e5_B()
t5Btot = time.time()
t5B = t5Btot - t5Atot
print("5_B =", e5_B,"In", t5B/60, "minutes (", t5B/3600, "hours )")

e6_A = e6_A()
t6Atot = time.time()
t6A = t6Atot - t5Btot
print("6_A =", e6_A, "In", t6A/60, "minutes (", t6A/3600, "hours )")

e6_B = e6_B()
t6Btot = time.time()
t6B = t6Btot - t6Atot
print("6_B =", e6_B, "In", t6B/60, "minutes (", t6B/3600, "hours )")

e7_A = e7_A()
t7Atot = time.time()
t7A = t7Atot - t6Btot
print("7_A =", e7_A, "In", t7A/60, "minutes (", t7A/3600, "hours )")

e7_B = e7_B()
t7Btot = time.time()
t7B = t7Btot - t7Atot
print("7_B =", e7_B, "In", t7B/60, "minutes (", t7B/3600, "hours )")

e8_A = e8_A()
t8Atot = time.time()
t8A = t8Atot -t7Btot
print("8_A =", e8_A, "In", t8A/60, "minutes (", t8A/3600, "hours )")

e8_B = e8_B()
t8Btot = time.time()
t8B = t8Btot - t8Atot
print("8_B =", e8_B, "In", t8B/60, "minutes (", t8B/3600, "hours )")


E = e1+e2+e3+e4+e5_A+e5_B+e6_A+e6_B+e7_A+e7_B+e8_A+e8_B
print("The configuration energy is", E)

tend = time.time()
t = tend - tstart
print("The energy took", t/60, "minutes to compute (", t/3600, "hours )")

