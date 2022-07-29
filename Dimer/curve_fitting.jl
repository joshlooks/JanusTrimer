using LsqFit

model(xdata,p) = p[1]*cos(p[2]*xdata)+p[2]*sin(p[1]*xdata)

xdata = [-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9]
ydata = [0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001]

fit = curve_fit(model, xdata, ydata, [1.0, 0.2])

beta = fit.param # best fit parameters
r = fit.resid # vector of residuals
J = fit.jacobian # estimated Jacobian at solution

#@printf(“Best fit parameters are: %f and %f”,beta[1],beta[2])
#@printf(“The sum of squares of residuals is %f”,sum(r.^2.0))