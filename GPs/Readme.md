

# Gaussian Processes With Python

A Gaussian Process (GP) is often denoted as:

$$
f(\mathbf{x}) \sim \mathcal{GP}\big(m(\mathbf{x}), K(\mathbf{x}, \mathbf{x}')\big)
$$

where:
- $m(\mathbf{x})$ is the **mean function**,
- $K(\mathbf{x}, \mathbf{x}')$ is the **kernel** (or covariance function), which defines the covariance matrix between any pair of inputs $\mathbf{x}$ and $\mathbf{x}'$.
<img width="859" height="545" alt="Untitled" src="https://github.com/user-attachments/assets/fe602c4a-2abe-4c43-bc22-4754f0927bda" />

<img width="900" height="500" alt="gp_animation" src="https://github.com/user-attachments/assets/61a57459-be75-4b76-924b-2a9141bfea85" />


# References: 

[MacKay, D. J. C. (2003). *Introduction to Gaussian Processes*.](https://www.inference.org.uk/mackay/gpB.pdf)

**Rasmussen & Williams (2006)**  
C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.  
[Online Book & PDF](https://gaussianprocess.org/gpml/)
