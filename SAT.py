import numpy
import scipy.integrate
import matplotlib.pyplot as plt
import pylab
import math

def rhs(W,T,beta,mu,sigma,phi,n,holder):
	X = W[0]
	Y = W[1:n+1]
	Y_j = numpy.zeros((n,n))
	for j in range(n):
		Y_j[:,j] = W[n*(j+1)+1:n*(j+2)+1]
	Z = W[n*(n+1)+1 : n*(n+2)+1]
	Z_star = W[-1]

	dx = mu - X*(beta*numpy.sum(Y + phi*numpy.sum(Y_j,axis=1))) - mu*X

	dy = X*(beta*(Y+phi*numpy.sum(Y_j,axis=1)))-sigma*Y - mu*Y

	dz = sigma*Y - numpy.sum(numpy.outer(Z,beta*(Y+phi*numpy.sum(Y_j,axis=1)))*holder,axis=1) - mu*Z

	dy_j = numpy.outer(Z,beta*(Y+phi*numpy.sum(Y_j,axis=1)))*holder - sigma*Y_j*holder - mu*Y_j*holder

	dz_star = sigma*numpy.sum(numpy.sum(Y_j*holder,axis=1)) - mu*Z_star

	dy_j = numpy.hstack(dy_j.T)


	dW = numpy.hstack((dx,dy,dy_j,dz,dz_star))
	return(dW)

def log_rhs(W,T,beta,mu,sigma,phi,n,holder):
	# is it acceptable to leave Z,Z_star,Y_j as non-transformed (I don't see why not)


	lnX = W[0]
	lnY = W[1:n+1]
	lnY_j = numpy.zeros((n,n))
	for j in range(n):
		lnY_j[:,j] = W[n*(j+1)+1:n*(j+2)+1]
	lnZ = W[n*(n+1)+1 : n*(n+2)+1]
	lnZ_star = W[-1]

	e = math.e

	dlnx = mu*e**-lnX - beta*numpy.sum(e**lnY+phi*numpy.sum(e**lnY_j,axis=1)) - mu



	#dlny = e**lnX*(beta*(1+phi*e**-lnY*numpy.sum(e**lnY_j*holder,axis=1)))- sigma - mu   # Original Line
	
	# Equation updated to combine exponential terms 

	# dy = X*(beta*(Y+phi*numpy.sum(Y_j,axis=1)))-sigma*Y - mu*Y
	
	dlny = beta*(e**lnX + phi*numpy.sum(e**(lnX+lnY_j-numpy.outer(lnY,numpy.ones(n)))*holder,axis=1)) - sigma - mu


	#dlnz = sigma*e**(lnY-lnZ) - e**-lnZ*numpy.sum(numpy.outer(e**lnZ,beta*(e**lnY+phi*numpy.sum(e**lnY_j,axis=1)))*holder,axis=1) - mu  # Original Line

	#Equation updated to combine exponential terms, where possible
	dlnz = sigma*e**(lnY-lnZ) - numpy.sum(numpy.outer(numpy.ones(n),beta*(e**lnY+phi*numpy.sum(e**lnY_j,axis=1)))*holder,axis=1) - mu

	dlny_j = e**-lnY_j*numpy.outer(e**lnZ,beta*(e**lnY+phi*numpy.sum(e**lnY_j*holder,axis=1)))*holder - sigma*holder - mu*holder

	#dlnz_star = e**-lnZ_star*sigma*numpy.sum(numpy.sum(e**lnY_j*holder,axis=1)) - mu # Original Line
	
	#Equation updated to combine exponential terms, where possible 
	dlnz_star = sigma*numpy.sum(numpy.sum(e**(lnY_j-lnZ_star)*holder,axis=1)) - mu  

	dlny_j = numpy.hstack(dlny_j.T)
	

	dW = numpy.hstack((dlnx,dlny,dlny_j,dlnz,dlnz_star))
	return(dW)

def run_SAT(p,phi,log=0,t_max=1000):
	if log==0:
		n = 4
		X = p #percentage initially susceptible
		Y = numpy.array([.25,.15,.075,.025])*(1-p)/.5
		Y_j = numpy.zeros(n*n)
		Z = numpy.zeros(n)
		Z_star = 0

		beta = 400
		mu = 0.02
		sigma = 100
		phi = phi

		t_max = t_max
		# X proportion susceptible
		# Y - vector of proportion infected with serotype i
		# Y_j - vector of proportion infected with serotype i who were previously infected with serotype j
		W0 = numpy.hstack((X,Y,Y_j,Z,Z_star))
		T = numpy.linspace(0,t_max,1000001)


# create a matrix of 0 on diag, 1's everywhere else to multiply by the outer product
		holder = 1 - numpy.eye(n)
		W = scipy.integrate.odeint(rhs,W0,T,args = (beta,mu,sigma,phi,n,holder))
		X,Y,Y_j,Z,Z_star = get_SIR_vals(W,T,n)

		Y_tot = Y + numpy.sum(Y_j,axis=1)


		plt.plot(T,Y_tot)
		plt.yscale('log')
		#plt.xlim((t_max-200,t_max))
		val = numpy.ones(len(T))
		plt.plot(T,1*10**-8*val,'--')
		Total = X + numpy.sum(Y_tot,axis=1) + numpy.sum(Z,axis=1) + Z_star
		pylab.show()
		return(W)
	elif log==1:
		n = 4
		X = numpy.log(p)
		Y = numpy.log(numpy.array([.25,.15,.075,.025])*(1-p)/.5)
		Y_j = -20*numpy.ones(n*n)
		Z = -20*numpy.ones(n)
		Z_star = -20

		beta = 400
		mu = 0.02
		sigma = 100
		phi = phi
		t_max = t_max
		# X proportion susceptible
		# Y - vector of proportion infected with serotype i
		# Y_j - vector of proportion infected with serotype i who were previously infected with serotype j
		W0 = numpy.hstack((X,Y,Y_j,Z,Z_star))
		T = numpy.linspace(0,t_max,1000001)
		holder = 1 - numpy.eye(n)
		log_W = scipy.integrate.odeint(log_rhs,W0,T,args = (beta,mu,sigma,phi,n,holder))
		log_X,log_Y,log_Y_j,log_Z,log_Z_star = get_SIR_vals(log_W,T,n)

		X = math.e**log_X
		Y = math.e**log_Y
		Y_j = math.e**log_Y_j
		Z = math.e**log_Z
		Z_star = math.e**log_Z_star

		Y_tot = Y + numpy.sum(Y_j,axis=1)

	
		plt.plot(T,Y_tot)
		plt.yscale('log')
		#plt.xlim((t_max-200,t_max))
		val = numpy.ones(len(T))
		plt.plot(T,1*10**-8*val,'--')
		Total = X + numpy.sum(Y_tot,axis=1) + numpy.sum(Z,axis=1) + Z_star
		pylab.show()
		return(log_W)

def get_SIR_vals(W,T,n):
	X = W[:,0]
	Y = W[:,1:n+1]
	Y_j = numpy.zeros((len(T),n,n))
	for j in range(n):
		Y_j[:,j] = W[:,n*(j+1)+1:n*(j+2)+1]
	Z = W[:,n*(n+1)+1 : n*(n+2)+1]
	Z_star = W[:,-1]
	return(X,Y,Y_j,Z,Z_star)

def pull_deriv_vals(W,n=4):
	dX = W[0]
	dY = W[1:n+1]
	dY_j = numpy.zeros((n,n))
	for j in range(n):
		dY_j[:,j] = W[n*(j+1)+1:n*(j+2)+1]
	dZ = W[n*(n+1)+1 : n*(n+2)+1]
	dZ_star = W[-1]
	return(dX,dY,dY_j,dZ,dZ_star)

# This function will run both ODE's, and compare the values at time t
def comparison_test(p=.1,phi=1.5,tol = 0.01):
	n = 4
	X = p #percentage initially susceptible
	ln_X = numpy.log(p)
	Y = numpy.array([.25,.15,.075,.025])*(1-p)/.5
	ln_Y = numpy.log(numpy.array([.25,.15,.075,.025])*(1-p)/.5)

	Y_j = numpy.exp(-20)*numpy.ones(n*n)
	ln_Y_j = -20*numpy.ones(n*n)

	Z = numpy.exp(-20)*numpy.ones(n)
	ln_Z = -20*numpy.ones(n)
	
	Z_star = numpy.exp(-20)
	ln_Z_star = -20

	beta = 400
	mu = 0.02
	sigma = 100
	phi = phi

	t_max = 1000
	holder = 1 - numpy.eye(n)
	W0 = numpy.hstack((X,Y,Y_j,Z,Z_star))
	ln_W0 = numpy.hstack((ln_X,ln_Y,ln_Y_j,ln_Z,ln_Z_star))
	dW_0 = rhs(W0,0,beta,mu,sigma,phi,n,holder)
	dln_W0 = log_rhs(ln_W0,0,beta,mu,sigma,phi,n,holder)
	dX,dY,dY_j,dZ,dZ_star = pull_deriv_vals(dW_0)
	dln_X,dln_Y,dln_Y_j,dln_Z,dln_Z_star = pull_deriv_vals(dln_W0)

	# We change the dimensions of the initial Y_j and ln_Y_j to allow for easier calculations below
	Y_j = numpy.exp(-20)*numpy.ones((n,n))
	ln_Y_j = -20*numpy.ones((n,n))

	tol = tol
	err_count = 0
	if numpy.abs(dln_X - dX/X)/dln_X>tol:
		print('Discrepency in dX values at time %d' %t)
		err_count+=1
	if numpy.max(numpy.abs((dln_Y-dY/Y)/dln_Y))>tol:
		print('Discrepency in dY values')
		err_count+=1
	if numpy.max(numpy.abs((dln_Y_j - dY_j/numpy.exp(ln_Y_j))))>tol:
		print('Discrepency in dY_j values')
		err_count+=1
	if numpy.max(numpy.abs(dln_Z - dZ/numpy.exp(ln_Z))/dln_Z)>tol:		#Roundoff error concerns here (equal for 8 siginifcal digits, but values are 2e10)
		print('Discrepency in dZ values')
		err_count+=1
	if numpy.abs((dln_Z_star - dZ_star/numpy.exp(ln_Z_star))/dln_Z_star)>tol:
		print('Discrepency in dZ_star values')
		err_count+=1
	if err_count == 0:
		print('No errors found in initial run')

	return()