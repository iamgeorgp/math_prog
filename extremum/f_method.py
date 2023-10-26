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

#--------------------------------------------------------------------------------------------------------
def newton_method(f, x, variables, epsilon):
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
def fast_descent(f, x, variables, tolerance, learning_rate):
	grad_f = [sympy.diff(f, var) for var in variables]
	x1 = variables[0]
	x2 = variables[1]
	x3 = variables[2]
	x_values = {x1: x[0], x2: x[1], x3: x[2]}
	iteration=0
	while True:
		iteration +=1
		# Вычисляем градиент в текущей точке
		gradient_at_point = [grad.subs(x_values) for grad in grad_f]
		
		# Проверяем условие останова
		if all(abs(gradient) < tolerance for gradient in gradient_at_point):
			break
		
		# Обновляем переменные
		for var, grad_value in zip((x1, x2, x3), gradient_at_point):
			x_values[var] -= learning_rate * grad_value
	
	# Результат
	x_result = [x_values[var] for var in variables]
	return x_result, iteration