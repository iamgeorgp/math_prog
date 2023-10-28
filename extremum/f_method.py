import sympy

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
#--------------------------------------------------------------------------------------------------------
def newton_method(f, x, variables, epsilon=0.001):
    # def gradient and Hesse matrix
    grad_f = [sympy.diff(f, var) for var in variables]
    Hessian_matrix = sympy.Matrix([[sympy.diff(grad_f[i], var) for var in variables] for i in range(len(variables))])
    # iterative formula of Newton's method
    x_current = sympy.Matrix(x)
    iteration = 0
    while True:
        gradient_value = [grad.subs({var: val for var, val in zip(variables, x_current)}) for grad in grad_f]
        Hessian_value = Hessian_matrix.subs({var: val for var, val in zip(variables, x_current)})
        delta_x = Hessian_value.inv() * sympy.Matrix(gradient_value)
        x_new = x_current - delta_x
        norm_delta_x = sympy.sqrt(sum(val**2 for val in delta_x))

        if norm_delta_x < epsilon:
            break

        x_current = x_new
        iteration += 1
    x_result = [float(each) for each in x_current]
    return x_result, iteration
#--------------------------------------------------------------------------------------------------------
def fast_descent(f, x, variables, tolerance=1e-6, epsilon=0.001):
	grad_f = [sympy.diff(f, var) for var in variables]
	x1 = variables[0]
	x2 = variables[1]
	x3 = variables[2]
	x_values = {x1: x[0], x2: x[1], x3: x[2]}
	iteration=0
	while True:
		gradient_at_point = [grad.subs(x_values) for grad in grad_f]
		
		if all(abs(gradient) < tolerance for gradient in gradient_at_point):
			break
		
		for var, grad_value in zip((x1, x2, x3), gradient_at_point):
			x_values[var] -= epsilon * grad_value
		iteration +=1

	x_result = [x_values[var] for var in variables]
	return x_result, iteration
#--------------------------------------------------------------------------------------------------------
def coordinate_descent(f, x, variables, epsilon, precision=1e-6):
    x1_val = x[0]; x1 = variables[0];
    x2_val = x[1]; x2 = variables[1];
    x3_val = x[2]; x3 = variables[2];
    iteration = 0
    while True:
        if iteration > 0:
            delta_x1 = abs(x1_val - x1_prev)
            delta_x2 = abs(x2_val - x2_prev)
            delta_x3 = abs(x3_val - x3_prev)
            if delta_x1 < precision and delta_x2 < precision and delta_x3 < precision:
                break
        
        x1_prev = x1_val
        x2_prev = x2_val
        x3_prev = x3_val
        
        df_dx1 = sympy.diff(f, x1).subs({x1: x1_val, x2: x2_val, x3: x3_val})
        df_dx2 = sympy.diff(f, x2).subs({x1: x1_val, x2: x2_val, x3: x3_val})
        df_dx3 = sympy.diff(f, x3).subs({x1: x1_val, x2: x2_val, x3: x3_val})
        
        x1_val -= epsilon * df_dx1
        x2_val -= epsilon * df_dx2
        x3_val -= epsilon * df_dx3
        
        iteration += 1
    x_result=[x1_val, x2_val, x3_val]
    return x_result, iteration

