This repository contains the simulations based on the ODE describing the interactions between $n$~drones. The model is as follows:

$$
	\begin{aligned}
	&\dot{x}_i(t)=v_i(t),\quad\quad &i\in\{0,\ldots,n-1\},\\
 	&\dot{v}_i(t)= A_{i}(1-\frac{v_{i}(t)}{{V}_{i}}-\frac{v_{i}(t)}{\kappa}&,
	\end{aligned}
$$
 
with initial conditions
 
$$
	\begin{aligned}
	&0\leq x_{n-1}(t_0) \leq \ldots \leq x_1(t_0) \leq x_0(t_0),\\
	&0 \leq v_i(t_0) \leq V_i,\quad\quad\quad i\in\{0,\ldots,n-1\}.
	\end{aligned}
$$

The variable $x_i(t)$ (expressed in meters ($m$)) describes the position of $i$-th drone (ordered from the drone at the front at time $t$, while variable $v_i$ – its velocity (expressed in $\frac{m}{s}$). Parameters $A_i$ and $V_i$ describe the maximal acceleration and the maximal velocity of the $i$-th drone, respectively. Parameters $m_i$ and $K_i$ describe the size of the $i$-th drone: $m_i$ is its mass and $K_i$ is proportional to the surface of its cross-section. Parameter $\kappa$ describes the capacity of the air corridor inside the horizon $\omega$, i.e. the distance in front of the drone, in which the preceding drones have a higher impact on the movement, while $H_i$ describes the wind force.
According to Battista et al. [1], the wind force acting on a drone can be described by:

$$
H_i(t)=\pm \frac{1}{2}\rho C_{d}A_{f,i}v_{wind}^{2}(t),
$$

where $\rho$ is a mass density of air, $C_{d}$ is a drag coefficient (dimensionless), $A_{f,i}$ is a frontal area of drone $i$ and $v_{wind}$ is a wind velocity.
Using formula (\ref{windforceliterature}), one can compute wind-induced acceleration acting on drone $i$ simply by dividing it by the mass of a drone. 
It is assumed that all parameters are positive.

[1] A. Battista, D. Ni (2017). Modeling Small Unmanned Aircraft System Traffic Flow Under External Force. Transportation Research Record: Journal of the Transportation Research Board. DOI:10.3141/2626-10.
