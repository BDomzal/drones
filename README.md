# Autonomous Drone Model

<img width="462" height="99" alt="fig1_alt-1" src="https://github.com/user-attachments/assets/63d04fea-59e8-4af7-b973-aedaad5d8a23" />

This repository contains the simulations based on the ODE describing the interactions between $n$ drones. The model is as follows:

$`
    \begin{aligned}
        \dot{x}_i(t) &= v_i(t),                                  &&i\in \{0, \dots, n-1\}\\
        \dot{v}_i(t) &= A_{i}\left(1-\frac{v_{i}(t)}{{V}_{i}}-v_{i}(t)S_i(t)\right)+\frac{H_i(t)}{m_i}, \ && i\in \{0, \dots, n-1\} \\
        S_i(t)       &={\Huge\{}\begin{aligned} & 0, \\ & \textstyle{\frac{1}{\kappa}\sum_{j=0}^{i-1} K_{j}\exp{\frac{x_i(t)-x_j(t)}{\omega}}}, 
		\end{aligned} &&
		\begin{aligned} 
		&i=0 \\ 
		&i\in \{1, \dots, n-1\}
		\end{aligned}
		\end{aligned}
`$
 
with initial conditions
 
$`
	\begin{aligned}
	&0\leq x_{n-1}(t_0) \leq \ldots \leq x_1(t_0) \leq x_0(t_0),\\
	&0 \leq v_i(t_0) \leq V_i,\quad\quad\quad i\in\{0,\ldots,n-1\}.
	\end{aligned}
`$

The variable $x_i(t)$ (expressed in meters [m]) describes the position of $i$-th drone (ordered from the drone at the front at time $t$, while variable $v_i$ – its velocity (expressed in $\frac{\text{m}}{\text{s}}$). Parameters $A_i$ and $V_i$ describe the maximum acceleration and the maximum velocity of the $i$-th drone, respectively. Parameters $m_i$ and $K_i$ describe the size of the $i$-th drone: $m_i$ is its mass and $K_i$ is the surface of its cross-section. Parameter $\kappa$ describes the capacity of the air corridor inside the horizon $\omega$, i.e. the distance in front of the drone, in which the preceding drones have a higher impact on the movement, while $H_i$ describes the wind force.
According to Battista et al. [1], the wind force acting on a drone can be described by:

$`
H_i(t)=\pm \frac{1}{2}\rho C_{d}A_{f,i}v_{wind}^{2}(t),
`$

where $\rho$ is a density of air, $C_{d}$ is a drag coefficient (dimensionless), $A_{f,i}$ is a frontal area of drone $i$ and $v_{wind}$ is a wind velocity.
Using formula above, one can compute wind-induced acceleration acting on drone $i$ simply by dividing it by the mass of a drone, in accordance with Newton's second law of motion. The parameters and units are summarized in the table below. It is assumed that all parameters are positive.

<img width="485" height="263" alt="Screenshot from 2025-12-06 12-08-18" src="https://github.com/user-attachments/assets/e33cf50a-b38f-4aa3-a27a-3f2f185109b6" />


[1] A. Battista, D. Ni (2017). Modeling Small Unmanned Aircraft System Traffic Flow Under External Force. Transportation Research Record: Journal of the Transportation Research Board. DOI:10.3141/2626-10.
