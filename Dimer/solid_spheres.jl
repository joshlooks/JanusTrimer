using LinearAlgebra
using HCubature


#----------------------------------------------------------------------------------------------------
# Important Functions

function cart(theta1, phi1, rp, thetap, phip)::Array{Float64,1}
    r = Float64[0,0,0]

    st1 = sin(theta1)
    ast1 = abs(st1)
    ct1 = cos(theta1)
    sp1 = sin(phi1)
    cp1 = cos(phi1)
    stp = sin(thetap)
    ctp = cos(thetap)
    spp = sin(phip)
    cpp = cos(phip)

    r[1] = rp*stp*cpp*st1*sp1/ast1 + rp*stp*spp*st1*ct1*cp1/ast1 + rp*ctp*st1*cp1
    r[2] = -rp*stp*cpp*st1*cp1/ast1 + rp*stp*spp*st1*ct1*sp1/ast1 + rp*ctp*st1*sp1
    r[3] =  -rp*stp*spp*ast1 + rp*ctp*ct1

    return r
end


function rhomo(rp, thetap, phip, rpp, thetapp, phipp)::Float64
    pos1p = Float64[0,0,0]
    pos1p[1] = rp*sin(thetap)*cos(phip)
    pos1p[2] = rp*sin(thetap)*sin(phip)
    pos1p[3] = rp*cos(thetap)

    pos2p = Float64[0,0,0]
    pos2p[1] = rpp*sin(thetapp)*cos(phipp)
    pos2p[2] = rpp*sin(thetapp)*sin(phipp)
    pos2p[3] = rpp*cos(thetapp)

    return convert(Float64, norm(pos1p-pos2p))
end


function rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, rpp, thetapp, phipp, x, y, z)::Float64
    pos1 = cart(theta1, phi1, rp, thetap, phip)
    pos2 = cart(theta2, phi2, rpp, thetapp, phipp) + [x,y,z]
    return convert(Float64, norm(pos1-pos2))
end


function rljp(sigma, r)::Float64
    x = (sigma/r)^6
    return convert(Float64, x^2-x)
end


#----------------------------------------------------------------------------------------------------
# Parameters

# LJ units
const epsilon = 1.00  
const sigma = 1.00

const A = 4*epsilon*sigma^6
const B = 4*epsilon*sigma^12

# LJ epsilon coefficients for different interactions
const aa = 1.00  
const ab = 0.05  
const bb = 0.0025  
const aw = 1.00  
const bw = 0.05  

# Distances
const R = 6.00*sigma  
const ds = 0.51*sigma
const dw = 0.80*sigma

# Atom density
const surfaceeta = 1.30/sigma^2
const volumeeta = 0.77/sigma^3

# Sphere Orientations

const theta1 = 180.01/180*pi
const phi1 = 0.00
const theta2 = 0.01/180*pi
const phi2 = 0.00/180*pi


#-----------------------------------------------------------------------------------------
# Integrands

function integrand1(x)
    phipp = x[1]
    thetapp = x[2]
    rpp = x[3]
    phip = x[4]
    thetap = x[5]
    rp = x[6]
    r = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, rpp, thetapp, phipp, 0.00, 0.00, -(2*R+ds))
    Vaa = rljp(sigma, r)
    return Vaa*rp^2*rpp^2*sin(thetap)*sin(thetapp)
end


function integrand2(x)
    phipp = x[1]
    thetapp = x[2]
    rpp = x[3]
    phip = x[4]
    thetap = x[5]
    rp = x[6]
    r = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, rpp, thetapp, phipp, 0.00, 0.00, -(2*R+ds))
    Vab = rljp(sigma, r)
    return Vab*rp^2*rpp^2*sin(thetap)*sin(thetapp)
end


function integrand3(x)
    phipp = x[1]
    thetapp = x[2]
    rpp = x[3]
    phip = x[4]
    thetap = x[5]
    rp = x[6]
    r = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, rpp, thetapp, phipp, 0.00, 0.00, -(2*R+ds))
    Vab = rljp(sigma, r)
    return Vab*rp^2*rpp^2*sin(thetap)*sin(thetapp)
end


function integrand4(x)
    phipp = x[1]
    thetapp = x[2]
    rpp = x[3]
    phip = x[4]
    thetap = x[5]
    rp = x[6]
    r = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, rpp, thetapp, phipp, 0.00, 0.00, -(2*R+ds))
    Vbb = rljp(sigma, r)
    return Vbb*rp^2*rpp^2*sin(thetap)*sin(thetapp)
end


function integrand5_B(x)::Float64
    a = x[1]
    phip = x[2]
    thetap = x[3]
    rp = x[4]
    delta = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, 0.00, 0.00, 0.00, 0.00, 0.00, -(2*R+ds))
    return -a/delta*(-A/2*(1/(delta-a)^4-1/(delta+a)^4)+B/5*(1/(delta-a)^10-1/(delta+a)^10))*rp^2*sin(thetap)
end


function integrand6_B(x)::Float64
    a = x[1]
    phip = x[2]
    thetap = x[3]
    rp = x[4]
    delta = rhetero(theta1, phi1, rp, thetap, phip, theta2, phi2, 0.00, 0.00, 0.00, 0.00, 0.00, -(2*R+ds))
    return -a/delta*(-A/2*(1/(delta-a)^4-1/(delta+a)^4)+B/5*(1/(delta-a)^10-1/(delta+a)^10))*rp^2*sin(thetap)
end


function integrand7_B(x)::Float64
    a = x[1]
    phipp = x[2]
    thetapp = x[3]
    rpp = x[4]
    delta = rhetero(theta1, phi1, 0.00, 0.00, 0.00, theta2, phi2, rpp, thetapp, phipp, 0.00, 0.00, -(2*R+ds))
    return -a/delta*(-A/2*(1/(delta-a)^4-1/(delta+a)^4)+B/5*(1/(delta-a)^10-1/(delta+a)^10))*rpp^2*sin(thetapp)
end


function integrand8_B(x)::Float64
    a = x[1]
    phipp = x[2]
    thetapp = x[3]
    rpp = x[4]
    delta = rhetero(theta1, phi1, 0.00, 0.00, 0.00, theta2, phi2, rpp, thetapp, phipp, 0.00, 0.00, -(2*R+ds))
    return -a/delta*(-A/2*(1/(delta-a)^4-1/(delta+a)^4)+B/5*(1/(delta-a)^10-1/(delta+a)^10))*rpp^2*sin(thetapp)
end


#-----------------------------------------------------------------------------------------
# Calculation

tstart = time()
e1 = surfaceeta^2*4*epsilon*aa*hcubature(integrand1, [0.00, pi/2, R-1.00, 0.00, pi/2, R-1.00], [2*pi, pi, R, 2*pi, pi, R], atol=1e-3)[1]
t1tot = time()
t1 = t1tot - tstart
println("1 = ", e1, " In ", t1/60, " minutes (", t1/3600, " hours)")


e2 = surfaceeta^2*4*epsilon*ab*hcubature(integrand2, [0.00, 0.00, R-1.00, 0.00, pi/2, R-1.00], [2*pi, pi/2, R, 2*pi, pi, R], atol=1e-3)[1]
t2tot = time()
t2 = t2tot - t1tot
println("2 = ", e2, " In ", t2/60, " minutes (", t2/3600, " hours)")


e3 = surfaceeta^2*4*epsilon*ab*hcubature(integrand3, [0.00, pi/2, R-1.00, 0.00, 0.00, R-1.00], [2*pi, pi, R, 2*pi, pi/2, R], atol=1e-3)[1]
t3tot = time()
t3 = t3tot - t2tot
println("3 = ", e3, " In ", t3/60, " minutes (", t3/3600, " hours)")


e4 = surfaceeta^2*4*epsilon*bb*hcubature(integrand4, [0.00, 0.00, R-1.00, 0.00, 0.00, R-1.00], [2*pi, pi/2, R, 2*pi, pi/2, R], atol=1e-3)[1]
t4tot = time()
t4 = t4tot - t3tot
println("4 = ", e4, " In ", t4/60, " minutes (", t4/3600, " hours)")


e5_B = pi*surfaceeta^2*volumeeta^2*aw*hcubature(integrand5_B, [0.00, 0.00, pi/2, R-1.00], [R, 2*pi, pi, R], atol=1e-3)[1]
t5tot = time()
t5 = t5tot - t4tot
println("5_B = ", e5_B, " In ", t5/60, " minutes (", t5/3600, " hours)")


e6_B = pi*surfaceeta^2*volumeeta^2*bw*hcubature(integrand6_B, [0.00, 0.00, 0.00, R-1.00], [R, 2*pi, pi/2, R], atol=1e-6)[1]
t6tot = time()
t6 = t6tot - t5tot
println("6_B = ", e6_B, " In ", t6/60, " minutes (", t6/3600, " hours)")


e7_B = pi*surfaceeta^2*volumeeta^2*aw*hcubature(integrand7_B, [0.00, 0.00, pi/2, R-1.00], [R, 2*pi, pi, R], atol=1e-6)[1]
t7tot = time()
t7 = t7tot - t6tot
println("7_B = ", e7_B, " In ", t7/60, " minutes (", t7/3600, " hours)")


e8_B = pi*surfaceeta^2*volumeeta^2*bw*hcubature(integrand8_B, [0.00, 0.00, 0.00, R-1.00], [R, 2*pi, pi/2, R], atol=1e-6)[1]
t8tot = time()
t8 = t8tot - t7tot
println("8_B = ", e8_B, " In ", t8/60, " minutes (", t8/3600, " hours)")


E = e1+e2+e3+e4+e5_B+e6_B+e7_B+e8_B
println("The configuration energy is ", E)

tend = time()
t = tend - tstart
println("The energy took ", t/60, " minutes to compute (", t/3600, " hours)")

