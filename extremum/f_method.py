#-------------------------------------------------------------------------------------------------------
import sympy
import scipy

#--------------------------------------------------------------------------------------------------------
def gradient_method_step_split(f, x, variables, alpha, beta, epsilon):
	# def gradient of function
	grad_f = [sympy.diff(f, var) for var in variables]
	# iteration of method
	iteration = 0
	x1 = variables[0]
	x2 = variables[1]
	x3 = variables[2]
	while True:
		gradient_value = [grad.subs({x1: x[0], x2: x[1], x3: x[2]}) for grad in grad_f]
		norm_gradient = sympy.sqrt(sum(val**2 for val in gradient_value))
		if norm_gradient < epsilon:
			break
		x_new = [x[i] - alpha * gradient_value[i] for i in range(3)]
		while f.subs({x1: x_new[0], x2: x_new[1], x3: x_new[2]}) > f.subs({x1: x[0], x2: x[1], x3: x[2]}):
			alpha *= beta
			x_new = [x[i] - alpha * gradient_value[i] for i in range(3)]
		x = x_new
		iteration += 1
	return iteration, x

#-------------------------------------------------------------------------------------------------------
def gradient_descent(f, x, variables, tol=1e-5):
    iteration = 0
    x1,x2,x3 = variables
    grad_f = [sympy.diff(f, var) for var in variables]
    while True:
        grad = [g.evalf(subs={x1: x[0], x2: x[1], x3: x[2]}) for g in grad_f]
        lambda_min = sympy.symbols('lambda')
        f_lambda = f.subs({x1: x[0] - lambda_min*grad[0], 
                           x2: x[1] - lambda_min*grad[1], 
                           x3: x[2] - lambda_min*grad[2]})
        lambda_min = sympy.solve(sympy.diff(f_lambda, lambda_min))[0]
        x = [x[j] - lambda_min*grad[j] for j in range(len(x))]
        if all(abs(g) < tol for g in grad):
            break
        iteration += 1
    return x, iteration

#-------------------------------------------------------------------------------------------------------
def newton_method(f, x, variables, epsilon=0.001):
    # def gradient and Hesse matrix
    grad_f = [sympy.diff(f, var) for var in variables]
    Hessian_matrix = sympy.Matrix([[sympy.diff(grad_f[i], var) for var in variables] for i in range(len(variables))])
    # iterative formula of Newton's method
    x_current = sympy.Matrix(x)
    iteration = 0
    while True:
        iteration += 1
        gradient_value = [grad.subs({var: val for var, val in zip(variables, x_current)}) for grad in grad_f]
        Hessian_value = Hessian_matrix.subs({var: val for var, val in zip(variables, x_current)})
        delta_x = Hessian_value.inv() * sympy.Matrix(gradient_value)
        x_new = x_current - delta_x
        norm_delta_x = sympy.sqrt(sum(val**2 for val in delta_x))
        if norm_delta_x < epsilon:
            break
        x_current = x_new
    x_result = [float(each) for each in x_current]
    return x_result, iteration

#-------------------------------------------------------------------------------------------------------
def conjugate_gradient_method(f, x, variables):
    x1 = variables[0]
    x2 = variables[1]
    x3 = variables[2]
    def func(x):
        return f.subs({x1: x[0], 
					   x2: x[1], 
					   x3: x[2]})

    res = scipy.optimize.minimize(
				   func, 
                   x, 
                   method='CG', 
                   options={'disp': False})
    x_point = res.x
    iteration = res.nit
    return x_point, iteration

#-------------------------------------------------------------------------------------------------------
def coordinate_descent_method(f, x0, variables, tolerance = 1e-6):
    x1 = variables[0]
    x2 = variables[1]
    x3 = variables[2]
    iteration = 0
    while True:
        f_prev = f.subs({xj: x0[j] for j, xj in enumerate([x1, x2, x3])})
        for i, xi in enumerate([x1, x2, x3]):
            f_xi = f.subs({xj: x0[j] for j, xj in enumerate([x1, x2, x3]) if j != i})
            result = scipy.optimize.minimize_scalar(lambda x: float(f_xi.subs(xi, x)), method='golden')  # Преобразуем результат к числу
            x0[i] = result.x
        f_current = f.subs({xj: x0[j] for j, xj in enumerate([x1, x2, x3])})
        iteration +=1
        if abs(f_current - f_prev) < tolerance:
            break
    return x0, iteration

#-------------------------------------------------------------------------------------------------------