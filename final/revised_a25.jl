using LinearAlgebra
using HCubature
using Distances

function create_angle_offset_data(angle_offset)
    sin1 = sin(angle_offset[1])
    cos1 = cos(angle_offset[1])
    sin2 = sin(angle_offset[2])
    cos2 = cos(angle_offset[2])

    asin1 = abs(sin1)

    sgn_sin1 = 1

    if sin1 < 0
        sgn_sin1 = -1
    end

    Float64[sin1, cos1, sin2, cos2, asin1, sgn_sin1]
end

# in terms of (r, theta, phi) - theta is polar angle
function convert_to_cartesian(spherical)::Array{Float64, 1}
    temp = sin(spherical[2])

    x = spherical[1] * cos(spherical[3]) * temp
    y = spherical[1] * sin(spherical[3]) * temp
    z = spherical[1] * cos(spherical[2])
    
    Float64[x, y, z]
end

function convert_to_cartesian_angle(spherical, angle_offset_data)::Array{Float64, 1}
    cartesian = Float64[0.0, 0.0, 0.0]

    # These are constant
    st1 = angle_offset_data[1]
    ast1 = angle_offset_data[5]

    test_sgn = angle_offset_data[6]

    # These are constant
    ct1 = angle_offset_data[2]
    sp1 = angle_offset_data[3]
    cp1 = angle_offset_data[4]

    stp = sin(spherical[2])
    ctp = cos(spherical[2])
    spp = sin(spherical[3])
    cpp = cos(spherical[3])

    stp_test_sgn = stp * test_sgn

    ctp_st1 = ctp * st1
    spp_ct1 = spp * ct1

    cartesian[1] = stp_test_sgn * (cpp*sp1 + spp_ct1*cp1) + ctp_st1*cp1
    cartesian[2] = stp_test_sgn * (spp_ct1*sp1 - cpp*cp1) + ctp_st1*sp1
    cartesian[3] =  -stp*spp*ast1 + ctp*ct1

    spherical[1] * cartesian
end

function distance_same_coodinates(spherical1, spherical2)::Float64
    cartesian1 = convert_to_cartesian(spherical1)
    cartesian2 = convert_to_cartesian(spherical2)
    euclidean(cartesian1, cartesian2)
end

function distance_same_coodinates_squared(spherical1, spherical2)::Float64
    cartesian1 = convert_to_cartesian(spherical1)
    cartesian2 = convert_to_cartesian(spherical2)
    sqeuclidean(cartesian1, cartesian2)
end

function distance_different_coordinates(spherical1, angle_offset1_data, spherical2, angle_offset2_data, translate2)::Float64
    cartesian1 = convert_to_cartesian_angle(spherical1, angle_offset1_data)
    cartesian2 = convert_to_cartesian_angle(spherical2, angle_offset2_data) + translate2
    euclidean(cartesian1, cartesian2)
end

function distance_different_coordinates_squared(spherical1, angle_offset1_data, spherical2, angle_offset2_data, translate2)::Float64
    cartesian1 = convert_to_cartesian_angle(spherical1, angle_offset1_data)
    cartesian2 = convert_to_cartesian_angle(spherical2, angle_offset2_data) + translate2
    sqeuclidean(cartesian1, cartesian2)
end

function potential(σ, r)::Float64
    temp = (σ/r)^6
    temp^2 - temp
end

function potential_distance_squared(σ2, r2)::Float64
    temp = (σ2/r2)^3
    temp^2 - temp
end

const array_index = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
# const array_index = 3

# LJ units
const ε = 1.00  
const σ = 1.00
const σ2 = σ^2
const A = 4*ε*σ^6
const B = 4*ε*σ^12

# LJ epsilon coefficients for different interactions
const aa = 1.00 # Hydrophillic - Hydrophillic
const ab = 0.25 # Hydrophillic - Hydrophobic
const bb = 0.0625 # Hydrophobic - Hydrophobic
const aw = 1.00 # Hydrophillic - Solvent
const bw = 0.25 # Hydrophobic - Solvent

# Distances
const R = 10.00*σ 
const R2 = R^2
const R4 = R^4 
const ds = 0.51*σ
const dw = 0.80*σ
const Rdw2 = (R + dw)^2

# Atom density
const surfaceeta = 1.30/σ^2
const volumeeta = 0.77/σ^3

# Sphere Orientations

global theta1s = range(2.5 * π/180, stop = 87.5 * π/180, length = 18)
global theta2s = range(92.5 * π/180, stop = 177.5 * π/180, length = 18)

# theta1 = theta1s[Int(round(task_index/19, RoundDown)) + 1]
# theta2 = theta2s[task_index % 19 + 1]

global theta1 = 15.0/180*π
global theta2 = (180.0-15.0)/180*π

global angle_offset1 = Float64[theta1, 0.0]
global angle_offset2 = Float64[theta2, 0.0]

global angle_offset1_data = create_angle_offset_data(angle_offset1)
global angle_offset2_data = create_angle_offset_data(angle_offset2)

angle_chooser = (array_index % 17)
phi_chooser = array_index ÷ 17

global partition_angle = (angle_chooser/18 + 1/18) * π
global phi = (phi_chooser / 9) * π

# global partition_angle = 3π / 18
# global phi = 0

const β = 2R + ds
const β2 = β^2

const translate2 = Float64[0, 0, β]

const maxevals1 = 12500000
const atol1 = 1e-8

const potential_coefficient = 4 * ε

# This function is for the integrand of each half of the particles
# interacting with another half of the other particle
# [theta1, phi1, theta2, phi2]
function sphere_sphere_integrand(x)::Float64
    spherical1 = Float64[R, x[1], x[2]]
    spherical2 = Float64[R, x[3], x[4]]

    r2 = distance_different_coordinates_squared(spherical1, angle_offset1_data, spherical2, angle_offset2_data, translate2)

    potential_distance_squared(σ2, r2) * R4 * sin(x[1]) * sin(x[3])
end

# Interaction between Hydrophillic sides
function e1()
    e = hcubature(sphere_sphere_integrand, Float64[0, 0, 0, 0], Float64[partition_angle, 2π, partition_angle, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    surfaceeta^2 * potential_coefficient * bb * e[1]
end

# Interaction between Hydrophillic and Hydrophobic
function e2()
    e = hcubature(sphere_sphere_integrand, Float64[partition_angle, 0, 0, 0], Float64[π, 2π, partition_angle, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    surfaceeta^2 * potential_coefficient * ab * e[1]
end

# Interaction between Hydrophillic and Hydrophobic
function e3()
    e = hcubature(sphere_sphere_integrand, Float64[0, 0, partition_angle, 0], Float64[partition_angle, 2π, π, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    surfaceeta^2 * potential_coefficient * ab * e[1]
end

# Interaction between the two Hydrophobic sides
function e4()
    e = hcubature(sphere_sphere_integrand, Float64[partition_angle, 0, partition_angle, 0], Float64[π, 2π, π, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    surfaceeta^2 * potential_coefficient * aa * e[1]
end

# This function calculates the integrand for the interaction between 
# a point on a sphere and the fluid in a spherical surface around it
# [r2, theta2, phi2]
function point_fluid_all_integrand(x)::Float64
    spherical1 = Float64[R, 0, 0]
    spherical2 = Float64[x[1], x[2], x[3]]

    r2 = distance_same_coodinates_squared(spherical1, spherical2)
    potential_distance_squared(σ2, r2) * x[1]^2 * sin(x[2])
end

# As the interaction is symmetric over the surface this multiplies by the surface area
# Calculates both interactions from both spheres
function e5()
    e = hcubature(point_fluid_all_integrand, Float64[R + dw, 0, 0], Float64[10000, π, 2π], maxevals = maxevals1)
    # println(point_e[1], " ", point_e[2])

    surface_area = 4 * π * R2 * (1 - cos(partition_angle))

    surfaceeta * volumeeta * potential_coefficient * bw * e[1] * surface_area
end

# Calculates both interactions for both spheres
function e6()
    e = hcubature(point_fluid_all_integrand, Float64[R + dw, 0, 0], Float64[10000, π, 2π], maxevals = maxevals1)
    # println(point_e[1], " ", point_e[2])

    surface_area = 4 * π * R2 * (1 + cos(partition_angle))

    surfaceeta * volumeeta * potential_coefficient * aw * e[1] * surface_area
end

# testing doing some spherical thing
# [theta1, phi1, R, z, phi]
function test_integrand1(x)::Float64
    value_thing = acos((x[3]^2 + β2 - Rdw2)/(2*β*x[3]))

    θ = x[4] * value_thing

    spherical1 = Float64[R, x[1], x[2]]
    spherical2 = Float64[x[3], θ, x[5]]

    test1 = convert_to_cartesian_angle(spherical1, angle_offset1_data)
    test2 = convert_to_cartesian(spherical2)

    r2 = sqeuclidean(test1, test2)

    potential_distance_squared(σ2, r2) * R2 * x[3]^2 * sin(x[1]) * sin(θ) * value_thing
end

function test_integrand2(x)::Float64
    value_thing = acos((x[3]^2 + β2 - Rdw2)/(2*β*x[3]))

    θ = x[4] * value_thing

    spherical1 = Float64[-x[3], θ, x[5]]
    spherical2 = Float64[R, x[1], x[2]]

    test1 = convert_to_cartesian_angle(spherical2, angle_offset2_data)
    test2 = convert_to_cartesian(spherical1)

    r2 = sqeuclidean(test1, test2)

    potential_distance_squared(σ2, r2) * R2 * x[3]^2 * sin(x[1]) * sin(θ) * value_thing
end

# Energy between bottom sphere first half and top sphere fluid volume
function e7()
    e = hcubature(test_integrand1, Float64[0, 0, R + dw, 0, 0], Float64[partition_angle, 2π, β + R + dw, 1, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    -surfaceeta * volumeeta * potential_coefficient * bw * e[1]
end

function e8()
    e = hcubature(test_integrand1, Float64[partition_angle, 0, R + dw, 0, 0], Float64[π, 2π, β + R + dw, 1, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    -surfaceeta * volumeeta * potential_coefficient * aw * e[1]
end

function e9()
    e = hcubature(test_integrand2, Float64[0, 0, R + dw, 0, 0], Float64[partition_angle, 2π, β + R + dw, 1, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    -surfaceeta * volumeeta * potential_coefficient * bw * e[1]
end

function e10()
    e = hcubature(test_integrand2, Float64[partition_angle, 0, R + dw, 0, 0], Float64[π, 2π, β + R + dw, 1, 2π], maxevals = maxevals1, atol = atol1)
    # println(e[1], " ", e[2])
    -surfaceeta * volumeeta * potential_coefficient * aw * e[1]
end

# array index goes from 0 to 18

# tasks = (19 * array_index):1:(19 * array_index + 18)
tasks = 0:1:323
# tasks = 18:1:18

global output = string()

for task_index in tasks
    global theta1 = theta1s[task_index ÷ 18 + 1]
    global theta2 = theta2s[task_index % 18 + 1]

    global angle_offset1 = Float64[theta1, 0.0]
    global angle_offset2 = Float64[theta2, phi]
    
    global angle_offset1_data = create_angle_offset_data(angle_offset1)
    global angle_offset2_data = create_angle_offset_data(angle_offset2)

    tstart = time()

    c1 = e1()
    t1 = time()
    println("E1: ", c1, " time: ", t1 - tstart, "s")

    c2 = e2()
    t2 = time()
    println("E2: ", c2, " time: ", t2 - t1, "s")

    c3 = e3()
    t3 = time()
    println("E3: ", c3, " time: ", t3 - t2, "s")

    c4 = e4()
    t4 = time()
    println("E4: ", c4, " time: ", t4 - t3, "s")

    c5 = e5()
    t5 = time()
    println("E5: ", c5, " time: ", t5 - t4, "s")

    c6 = e6()
    t6 = time()
    println("E6: ", c6, " time: ", t6 - t5, "s")

    c7 = e7()
    t7 = time()
    println("E7: ", c7, " time: ", t7 - t6, "s")

    c8 = e8()
    t8 = time()
    println("E8: ", c8, " time: ", t8 - t7, "s")

    c9 = e9()
    t9 = time()
    println("E9: ", c9, " time: ", t9 - t8, "s")

    c10 = e10()
    t10 = time()
    println("E10: ", c10, " time: ", t10 - t9, "s")

    et = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10
    println("ET: ", et, " time: ", t10 - tstart, "s")

    s = string(round(theta1 * 180/π, digits = 1), ",", round((π - theta2) * 180/π, digits = 1), ",", c1, ",", c2, ",", c3, ",", c4, ",", c5, ",", c6, ",", c7, ",", c8, ",", c9, ",", c10, ",", et, "\n")
    global output = string(output, s)
end

file = string("/nesi/nobackup/uoa00623/Jack/vary_patch_A25/J", angle_chooser + 1, "0_P", 2 * phi_chooser, "0.csv")
# # file = "/nesi/nobackup/uoa00623/Jack/integration_S051_W080_R10_P0_A025.csv"
# # file = "integration.csv"

# append the csv columns to the data
if !isfile(file)
    global output = string("theta1,theta2,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,energy\n", output)
end

f = open(file, "a+", lock = true)
println(f, output)
close(f)