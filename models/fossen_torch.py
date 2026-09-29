import matplotlib.pyplot as plt
import numpy as np
import torch

from run_learning_fossen_enhanced import Dataset, FossenNN
from run_learning_PINN import PINN


# Edit these values directly when running from the VS Code Run button.
DATA_DIR = "data"
SAMPLE_INDEX = 0
PREDICTION_STEPS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adapt_model_output_for_prediction(model_output):
	"""
	Normalize model output to prediction format [pos(3), R(9), vel(6), hydro(6)].

	Supports models that return either:
	- 24 channels (already compatible), or
	- 18 channels ([pos, R, vel]) where hydro channels are absent.
	"""
	if not torch.is_tensor(model_output):
		raise TypeError("model_output must be a torch.Tensor")

	if model_output.dim() == 1:
		model_output = model_output.unsqueeze(0)
		squeeze_back = True
	elif model_output.dim() == 2:
		squeeze_back = False
	else:
		raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(model_output.shape)}")

	channels = model_output.shape[1]
	if channels == 24:
		adapted = model_output
	elif channels == 18:
		zeros = torch.zeros(model_output.shape[0], 6, dtype=model_output.dtype, device=model_output.device)
		adapted = torch.cat((model_output, zeros), dim=1)
	else:
		raise ValueError(f"Unsupported model output width {channels}; expected 18 or 24")

	if squeeze_back:
		return adapted[0]
	return adapted

def skew_symmetric(vec):
	return np.array([
		[0.0, -vec[2], vec[1]],
		[vec[2], 0.0, -vec[0]],
		[-vec[1], vec[0], 0.0],
	])

def rodrigues_rotation(angular_increment):
	angle = np.linalg.norm(angular_increment)
	if angle < 1e-12:
		return np.eye(3) + skew_symmetric(angular_increment)

	axis = angular_increment / angle
	skew_axis = skew_symmetric(axis)
	return (
		np.eye(3)
		+ np.sin(angle) * skew_axis
		+ (1.0 - np.cos(angle)) * (skew_axis @ skew_axis)
	)


def gram_schmidt_orthonormalize(rotation_matrix):
	first_col = rotation_matrix[:, 0]
	first_col = first_col / np.linalg.norm(first_col)

	second_col = rotation_matrix[:, 1] - np.dot(rotation_matrix[:, 1], first_col) * first_col
	second_col = second_col / np.linalg.norm(second_col)

	third_col = np.cross(first_col, second_col)
	third_col = third_col / np.linalg.norm(third_col)

	return np.column_stack((first_col, second_col, third_col))


def integrate_rotation_matrix(rotation_matrix, angular_velocity_body, dt):
	rotation_update = rodrigues_rotation(angular_velocity_body * dt)
	next_rotation = rotation_matrix @ rotation_update
	# Re-orthonormalize to limit drift from repeated finite-precision updates.
	return gram_schmidt_orthonormalize(next_rotation)


def rotation_matrix_to_euler_radians(rotation_matrix):
	sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
	singular = sy < 1e-6

	if not singular:
		phi = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
		theta = np.arctan2(-rotation_matrix[2, 0], sy)
		psi = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
	else:
		phi = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
		theta = np.arctan2(-rotation_matrix[2, 0], sy)
		psi = 0.0

	return phi, theta, psi

def fossen_model(init_state, inputs, dt, parameters=None):
	# Rigid body & hydro parameters (Table III)
	m = 11.5
	V = 1.5
	rho = 1000.0
	L = 0.4
	force_scale = 0.5 * rho * V**2 * L**2
	torque_scale = 0.5 * rho * V**2 * L**3
	W_over_B = 0.98
	bouyancy = m * 9.81 * W_over_B
	r_g = [0.0, 0.0, 0.0]
	r_b = [0.0, 0.0, -0.02]
	I_x = 0.16
	I_y = 0.16
	I_z = 0.16
	
	current_state = np.array(init_state, dtype=float)
	rotation_matrix = np.eye(3)

	# Added-mass derivatives.
	X_u_dot = -5.5
	Y_v_dot = -12.7
	Z_w_dot = -14.57
	K_p_dot = -5.5
	M_q_dot = -5.5
	N_r_dot = -5.5

	# Linear damping
	X_u = -4.03
	Y_v = -6.22
	Z_w = -5.18
	K_p = -0.07
	M_q = -0.07
	N_r = -0.07

	# Quadratic damping 
	X_u_abs_u = -18.18
	Y_v_abs_v = -21.66
	Z_w_abs_w = -39.99
	K_p_abs_p = -1.55
	M_q_abs_q = -1.55
	N_r_abs_r = -1.55
	
	# If parameters are stated, override the default values with provided ones.
	if parameters is not None:
		X_u_dot = parameters[0]
		Y_v_dot = parameters[1]
		Z_w_dot = parameters[2]
		K_p_dot = parameters[3]
		M_q_dot = parameters[4]
		N_r_dot = parameters[5]
		X_u = parameters[6]
		Y_v = parameters[7]
		Z_w = parameters[8]
		K_p = parameters[9]
		M_q = parameters[10]
		N_r = parameters[11]
		X_u_abs_u = parameters[12]
		Y_v_abs_v = parameters[13]
		Z_w_abs_w = parameters[14]
		K_p_abs_p = parameters[15]
		M_q_abs_q = parameters[16]
		N_r_abs_r = parameters[17]	
		
	# Extract center of gravity
	x_g, y_g, z_g = r_g
	
	# Define prediction vector
	predictions = []

	# Loop over time steps and compute state derivatives using Fossen's equations of motion.
	for step_idx, step_input in enumerate(inputs):
		# Keep the baseline model numerically stable for long open-loop rollouts.
		current_state = np.nan_to_num(current_state, nan=0.0, posinf=50.0, neginf=-50.0)
		current_state = np.clip(current_state, -50.0, 50.0)
		u, v, w, p, q, r = current_state
		phi, theta, psi = rotation_matrix_to_euler_radians(rotation_matrix)

		# For coriolis matrix, we need to compute the terms that depend on the current velocities.
		a1 = X_u_dot * u
		a2 = Y_v_dot * v
		a3 = Z_w_dot * w
		b1 = K_p_dot * p
		b2 = M_q_dot * q
		b3 = N_r_dot * r
		
		# Rigid-body mass matrix
		M_RB = np.array([
			[m, 0, 0, 0, m*z_g, -m*y_g],
			[0, m, 0, -m*z_g, 0, m*x_g],
			[0, 0, m, m*y_g, -m*x_g, 0],
			[0, -m*z_g, m*y_g, I_x, 0, 0],
			[m*z_g, 0, -m*x_g, 0, I_y, 0],
			[-m*y_g, m*x_g, 0, 0, 0, I_z]
		])

		# Added mass matrix
		M_A = -1.0 * np.diag([
			X_u_dot,
			Y_v_dot,
			Z_w_dot,
			K_p_dot,
			M_q_dot,
			N_r_dot
		])
		
		# Coriolis and centripetal matrix (rigid body + added mass)
		C_RB = np.array([
			[0, 0, 0, 0,  m*w, -m*v],
			[0, 0, 0, -m*w, 0,  m*u],
			[0, 0, 0,  m*v, -m*u, 0],
			
			[0,  m*w, -m*v, 0,  I_z*r, -I_y*q],
			[-m*w, 0,  m*u, -I_z*r, 0,  I_x*p],
			[m*v, -m*u, 0,  I_y*q, -I_x*p, 0]
		])
		
		# Coriolis matrix from added mass
		C_A = np.array([
			[0, 0, 0, 0,  a3, -a2],
			[0, 0, 0, -a3, 0,  a1],
			[0, 0, 0,  a2, -a1, 0],
			[0,  a3, -a2, 0,  b3, -b2],
			[-a3, 0,  a1, -b3, 0,  b1],
			[a2, -a1, 0,  b2, -b1, 0]
		])

		# Linear damping matrix
		D_L = -1.0 * np.diag([
			X_u,
			Y_v,
			Z_w,
			K_p,
			M_q,
			N_r
		])

		# Nonlinear damping matrix (symbolic form using placeholders u,v,w,p,q,r)
		D_NL = -1.0 * np.array([
			[X_u_abs_u * abs(u), 0, 0, 0, 0, 0],
			[0, Y_v_abs_v * abs(v), 0, 0, 0, 0],
			[0, 0, Z_w_abs_w * abs(w), 0, 0, 0],
			[0, 0, 0, K_p_abs_p * abs(p), 0, 0],
			[0, 0, 0, 0, M_q_abs_q * abs(q), 0],
			[0, 0, 0, 0, 0, N_r_abs_r * abs(r)]
		])
		
		g_eta = np.array([
			0.0,
			0.0,
			0.0,
			bouyancy * (r_g[2] - r_b[2]) * np.sin(theta),
			-bouyancy * (r_g[2] - r_b[2]) * np.sin(phi) * np.cos(theta),
			0.0
		])
		
		# Define the total mass, coriolis, and damping matrices.
		M = M_RB + M_A
		C = C_RB + C_A
		D = D_L + D_NL
		
		# Input mapping:
		# - len==6: [Fx, Fy, Fz, Tx, Ty, Tz]
		# - len==12 (dataset): [state(6), Fx, Fy, Fz, Tx, Ty, Tz]
		if len(step_input) == 6:
			tau = np.concatenate((
				step_input[0:3],#* force_scale,
				step_input[3:6],# * torque_scale,
			))
		else:
			tau = np.concatenate((
				step_input[6:9],# * force_scale,
				step_input[9:12],# * torque_scale,
			))
		
		# Compute the acceleration using Fossen's equations of motion.
		state_dot = np.linalg.solve(M, tau - C @ current_state - D @ current_state - g_eta)
		state_dot = np.nan_to_num(state_dot, nan=0.0, posinf=50.0, neginf=-50.0)
		state_dot = np.clip(state_dot, -50.0, 50.0)
		
		# Integrate to get the new velocity (using simple Euler integration for demonstration).
		current_state = current_state + state_dot * dt
		rotation_matrix = integrate_rotation_matrix(rotation_matrix, current_state[3:6], dt)
		
		# Extract Euler angles for bouyancy term
		angles = rotation_matrix_to_euler_radians(rotation_matrix)
		theta, phi, psi = angles
		
		# Store the predicted state (velocity components) for this time step.
		predictions.append(current_state.copy())

	return np.array(predictions)
