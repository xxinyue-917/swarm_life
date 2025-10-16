# Particle Life Research

## Core Dynamics Equation

The fundamental equation governing particle motion in the system:

$$\ddot{\mathbf{x}}_i = \sum_j \frac{\hat{\mathbf{r}}}{r} F(r, a)$$

$$F(r, a) = a \cdot r \cdot (1 -
  \frac{r}{r_{max}})$$

where:
- $\ddot{\mathbf{x}}_i$ is the acceleration of particle $i$
- The summation is over all neighboring particles $j$
- $\hat{\mathbf{r}}$ is the unit vector pointing from particle $i$ to particle $j$
- $r$ is the distance between particles
- $F(r, a)$ is the force function with parameter $a$ determining the interaction strength