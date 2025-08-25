---
layout: default
title: "Mathematical Examples"
description: "Demonstration of MathJax rendering for mathematical content"
---

# Mathematical Content

This demonstrates inline math like $E = mc^2$ and $\alpha = \beta + \gamma$.

Here's display math for the Poisson equation:

$$\mathcal{L}u = f \quad \text{in } \Omega \subset \mathbb{R}^d$$

And the discrete Laplacian:
$$\mathcal{L}_h u_{i,j} = \frac{1}{h^2}[u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}]$$

Complex equation with cases:

$$\|\mathbf{e}^{(k)}\|_2 \leq \begin{cases} 
\rho_{\text{fp32}}^k \|\mathbf{e}^{(0)}\|_2 + C\epsilon_{32} & k < k_{\text{switch}} \\
\rho_{\text{fp64}}^{k-k_{\text{switch}}} \|\mathbf{e}^{(k_{\text{switch}})}\|_2 + C\epsilon_{64} & k \geq k_{\text{switch}}
\end{cases}$$

## V-Cycle Convergence Rate

The V-cycle convergence factor is bounded by:

$$\rho_V \leq \frac{2\rho_{TG}}{1 + \rho_{TG}} \quad \text{where} \quad \rho_{TG} = \|T_{TG}\|_{\mathcal{A}_h} < 1$$

## Mixed-Precision Speedup Formula

The theoretical speedup achievable with mixed precision is:

$$S_{\text{mixed}} = \frac{K \cdot C_{\text{fp64}}}{k_{\text{switch}} \cdot C_{\text{fp32}} + (K-k_{\text{switch}}) \cdot C_{\text{fp64}} + C_{\text{conversion}}}$$

## Matrix Notation

The restriction and prolongation operators satisfy:
$$I_{2h}^h : \mathcal{G}_h \to \mathcal{G}_{2h}, \quad I_h^{2h} : \mathcal{G}_{2h} \to \mathcal{G}_h$$

And the Galerkin condition:
$$\mathbf{A}_{2h} = I_{2h}^h \mathbf{A}_h I_h^{2h}$$

## Performance Analysis

The asymptotic complexity is:
$$\mathcal{C}_V(N) = \mathcal{O}(N) \sum_{k=0}^{L-1} \left(\frac{1}{2^d}\right)^k \approx \mathcal{O}(N)$$

where $N$ is the number of grid points and $d$ is the spatial dimension.