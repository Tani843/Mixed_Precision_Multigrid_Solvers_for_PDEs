"""CUDA kernels for GPU-accelerated multigrid operations."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class CUDAKernels:
    """
    Base class for CUDA kernel implementations.
    
    Provides common utilities and kernel management for GPU operations.
    """
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA kernels."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for CUDA kernels")
        
        self.device_id = device_id
        self.compiled_kernels: Dict[str, Any] = {}
        
        # Thread block configurations
        self.block_configs = {
            '1d': (256,),
            '2d_small': (16, 16),
            '2d_medium': (32, 16),
            '2d_large': (32, 32)
        }
        
        logger.debug(f"CUDA kernels initialized for device {device_id}")
    
    def get_optimal_block_size(self, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get optimal thread block size for given grid shape."""
        nx, ny = grid_shape
        
        if nx <= 64 and ny <= 64:
            return self.block_configs['2d_small']
        elif nx <= 512 and ny <= 512:
            return self.block_configs['2d_medium']
        else:
            return self.block_configs['2d_large']
    
    def get_grid_size(self, array_shape: Tuple[int, int], block_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate grid size for kernel launch."""
        return (
            (array_shape[0] + block_size[0] - 1) // block_size[0],
            (array_shape[1] + block_size[1] - 1) // block_size[1]
        )
    
    def compile_kernel(self, kernel_name: str, kernel_code: str) -> Any:
        """Compile and cache CUDA kernel."""
        if kernel_name in self.compiled_kernels:
            return self.compiled_kernels[kernel_name]
        
        try:
            kernel = cp.RawKernel(kernel_code, kernel_name)
            self.compiled_kernels[kernel_name] = kernel
            logger.debug(f"Compiled CUDA kernel: {kernel_name}")
            return kernel
        except Exception as e:
            logger.error(f"Failed to compile kernel {kernel_name}: {e}")
            raise


class SmoothingKernels(CUDAKernels):
    """CUDA kernels for smoothing operations (Jacobi, Gauss-Seidel)."""
    
    def __init__(self, device_id: int = 0):
        """Initialize smoothing kernels."""
        super().__init__(device_id)
        self._compile_smoothing_kernels()
    
    def _compile_smoothing_kernels(self) -> None:
        """Compile all smoothing kernels."""
        
        # Jacobi smoothing kernel
        jacobi_kernel_code = '''
        extern "C" __global__
        void jacobi_smoothing_kernel(
            const float* u_old,
            float* u_new,
            const float* rhs,
            const float hx_inv2,
            const float hy_inv2,
            const float center_coeff,
            const int nx,
            const int ny
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                int idx = i * ny + j;
                
                // 5-point stencil computation
                float stencil_result = hx_inv2 * (u_old[(i+1)*ny + j] + u_old[(i-1)*ny + j]) +
                                      hy_inv2 * (u_old[i*ny + (j+1)] + u_old[i*ny + (j-1)]) +
                                      center_coeff * u_old[idx];
                
                // Jacobi update
                u_new[idx] = (rhs[idx] - stencil_result) / (-center_coeff);
            }
        }
        '''
        
        # Double precision Jacobi kernel
        jacobi_kernel_code_double = '''
        extern "C" __global__
        void jacobi_smoothing_kernel_double(
            const double* u_old,
            double* u_new,
            const double* rhs,
            const double hx_inv2,
            const double hy_inv2,
            const double center_coeff,
            const int nx,
            const int ny
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                int idx = i * ny + j;
                
                // 5-point stencil computation
                double stencil_result = hx_inv2 * (u_old[(i+1)*ny + j] + u_old[(i-1)*ny + j]) +
                                       hy_inv2 * (u_old[i*ny + (j+1)] + u_old[i*ny + (j-1)]) +
                                       center_coeff * u_old[idx];
                
                // Jacobi update
                u_new[idx] = (rhs[idx] - stencil_result) / (-center_coeff);
            }
        }
        '''
        
        # Red-Black Gauss-Seidel kernel
        rb_gauss_seidel_kernel_code = '''
        extern "C" __global__
        void red_black_gauss_seidel_kernel(
            float* u,
            const float* rhs,
            const float hx_inv2,
            const float hy_inv2,
            const float center_coeff,
            const int nx,
            const int ny,
            const int color
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                // Red-black coloring: color = 0 for red, 1 for black
                if ((i + j) % 2 == color) {
                    int idx = i * ny + j;
                    
                    // 5-point stencil computation
                    float stencil_result = hx_inv2 * (u[(i+1)*ny + j] + u[(i-1)*ny + j]) +
                                          hy_inv2 * (u[i*ny + (j+1)] + u[i*ny + (j-1)]) +
                                          center_coeff * u[idx];
                    
                    // Gauss-Seidel update
                    u[idx] = (rhs[idx] - stencil_result) / (-center_coeff);
                }
            }
        }
        '''
        
        # SOR (Successive Over-Relaxation) kernel
        sor_kernel_code = '''
        extern "C" __global__
        void sor_kernel(
            float* u,
            const float* rhs,
            const float hx_inv2,
            const float hy_inv2,
            const float center_coeff,
            const float omega,
            const int nx,
            const int ny,
            const int color
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                if ((i + j) % 2 == color) {
                    int idx = i * ny + j;
                    
                    // 5-point stencil computation
                    float stencil_result = hx_inv2 * (u[(i+1)*ny + j] + u[(i-1)*ny + j]) +
                                          hy_inv2 * (u[i*ny + (j+1)] + u[i*ny + (j-1)]) +
                                          center_coeff * u[idx];
                    
                    // SOR update
                    float new_val = (rhs[idx] - stencil_result) / (-center_coeff);
                    u[idx] = (1.0f - omega) * u[idx] + omega * new_val;
                }
            }
        }
        '''
        
        # Block-based Jacobi kernel with shared memory
        jacobi_block_kernel_code = '''
        extern "C" __global__
        void jacobi_block_kernel(
            const float* u_old,
            float* u_new,
            const float* rhs,
            const float hx_inv2,
            const float hy_inv2,
            const float center_coeff,
            const int nx,
            const int ny
        ) {
            // Shared memory for block plus halo
            extern __shared__ float s_u[];
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockDim.x;
            int by = blockDim.y;
            
            int i = blockIdx.x * blockDim.x + tx;
            int j = blockIdx.y * blockDim.y + ty;
            
            // Load data to shared memory including halo
            int s_i = tx + 1;
            int s_j = ty + 1;
            int s_idx = s_i * (by + 2) + s_j;
            
            if (i < nx && j < ny) {
                s_u[s_idx] = u_old[i * ny + j];
                
                // Load halo regions
                if (tx == 0 && i > 0) {
                    s_u[(s_i-1) * (by + 2) + s_j] = u_old[(i-1) * ny + j];
                }
                if (tx == bx-1 && i < nx-1) {
                    s_u[(s_i+1) * (by + 2) + s_j] = u_old[(i+1) * ny + j];
                }
                if (ty == 0 && j > 0) {
                    s_u[s_i * (by + 2) + (s_j-1)] = u_old[i * ny + (j-1)];
                }
                if (ty == by-1 && j < ny-1) {
                    s_u[s_i * (by + 2) + (s_j+1)] = u_old[i * ny + (j+1)];
                }
            }
            
            __syncthreads();
            
            // Compute update using shared memory
            if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                int idx = i * ny + j;
                
                float stencil_result = hx_inv2 * (s_u[(s_i+1) * (by + 2) + s_j] + 
                                                  s_u[(s_i-1) * (by + 2) + s_j]) +
                                      hy_inv2 * (s_u[s_i * (by + 2) + (s_j+1)] + 
                                                 s_u[s_i * (by + 2) + (s_j-1)]) +
                                      center_coeff * s_u[s_idx];
                
                u_new[idx] = (rhs[idx] - stencil_result) / (-center_coeff);
            }
        }
        '''
        
        # Compile kernels
        self.compile_kernel("jacobi_smoothing_kernel", jacobi_kernel_code)
        self.compile_kernel("jacobi_smoothing_kernel_double", jacobi_kernel_code_double)
        self.compile_kernel("red_black_gauss_seidel_kernel", rb_gauss_seidel_kernel_code)
        self.compile_kernel("sor_kernel", sor_kernel_code)
        self.compile_kernel("jacobi_block_kernel", jacobi_block_kernel_code)
    
    def jacobi_smoothing(
        self,
        u_old: 'cp.ndarray',
        u_new: 'cp.ndarray',
        rhs: 'cp.ndarray',
        hx: float,
        hy: float,
        num_iterations: int = 1,
        use_shared_memory: bool = False
    ) -> None:
        """
        Perform Jacobi smoothing on GPU.
        
        Args:
            u_old: Input solution array
            u_new: Output solution array
            rhs: Right-hand side array
            hx: Grid spacing in x direction
            hy: Grid spacing in y direction
            num_iterations: Number of smoothing iterations
            use_shared_memory: Use shared memory optimization
        """
        nx, ny = u_old.shape
        hx_inv2 = 1.0 / (hx * hx)
        hy_inv2 = 1.0 / (hy * hy)
        center_coeff = -2.0 * (hx_inv2 + hy_inv2)
        
        # Get kernel based on data type and optimization
        if use_shared_memory:
            kernel = self.compiled_kernels["jacobi_block_kernel"]
        elif u_old.dtype == np.float64:
            kernel = self.compiled_kernels["jacobi_smoothing_kernel_double"]
        else:
            kernel = self.compiled_kernels["jacobi_smoothing_kernel"]
        
        # Configure kernel launch parameters
        block_size = self.get_optimal_block_size((nx, ny))
        grid_size = self.get_grid_size((nx, ny), block_size)
        
        # Shared memory size for block kernel
        if use_shared_memory:
            shared_mem_size = (block_size[0] + 2) * (block_size[1] + 2) * u_old.itemsize
        else:
            shared_mem_size = 0
        
        for iteration in range(num_iterations):
            if use_shared_memory:
                kernel(
                    grid_size, block_size,
                    (u_old, u_new, rhs, hx_inv2, hy_inv2, center_coeff, nx, ny),
                    shared_mem=shared_mem_size
                )
            else:
                kernel(
                    grid_size, block_size,
                    (u_old, u_new, rhs, hx_inv2, hy_inv2, center_coeff, nx, ny)
                )
            
            # Swap arrays for next iteration
            if iteration < num_iterations - 1:
                u_old, u_new = u_new, u_old
        
        cp.cuda.Device().synchronize()
    
    def red_black_gauss_seidel(
        self,
        u: 'cp.ndarray',
        rhs: 'cp.ndarray',
        hx: float,
        hy: float,
        num_iterations: int = 1
    ) -> None:
        """
        Perform Red-Black Gauss-Seidel smoothing on GPU.
        
        Args:
            u: Solution array (modified in-place)
            rhs: Right-hand side array
            hx: Grid spacing in x direction
            hy: Grid spacing in y direction
            num_iterations: Number of smoothing iterations
        """
        nx, ny = u.shape
        hx_inv2 = 1.0 / (hx * hx)
        hy_inv2 = 1.0 / (hy * hy)
        center_coeff = -2.0 * (hx_inv2 + hy_inv2)
        
        kernel = self.compiled_kernels["red_black_gauss_seidel_kernel"]
        
        # Configure kernel launch parameters
        block_size = self.get_optimal_block_size((nx, ny))
        grid_size = self.get_grid_size((nx, ny), block_size)
        
        for iteration in range(num_iterations):
            # Red points (color = 0)
            kernel(
                grid_size, block_size,
                (u, rhs, hx_inv2, hy_inv2, center_coeff, nx, ny, 0)
            )
            cp.cuda.Device().synchronize()
            
            # Black points (color = 1)
            kernel(
                grid_size, block_size,
                (u, rhs, hx_inv2, hy_inv2, center_coeff, nx, ny, 1)
            )
            cp.cuda.Device().synchronize()
    
    def sor_smoothing(
        self,
        u: 'cp.ndarray',
        rhs: 'cp.ndarray',
        hx: float,
        hy: float,
        omega: float = 1.0,
        num_iterations: int = 1
    ) -> None:
        """
        Perform SOR (Successive Over-Relaxation) smoothing on GPU.
        
        Args:
            u: Solution array (modified in-place)
            rhs: Right-hand side array
            hx: Grid spacing in x direction
            hy: Grid spacing in y direction
            omega: Relaxation parameter
            num_iterations: Number of smoothing iterations
        """
        nx, ny = u.shape
        hx_inv2 = 1.0 / (hx * hx)
        hy_inv2 = 1.0 / (hy * hy)
        center_coeff = -2.0 * (hx_inv2 + hy_inv2)
        
        kernel = self.compiled_kernels["sor_kernel"]
        
        # Configure kernel launch parameters
        block_size = self.get_optimal_block_size((nx, ny))
        grid_size = self.get_grid_size((nx, ny), block_size)
        
        for iteration in range(num_iterations):
            # Red points (color = 0)
            kernel(
                grid_size, block_size,
                (u, rhs, hx_inv2, hy_inv2, center_coeff, omega, nx, ny, 0)
            )
            cp.cuda.Device().synchronize()
            
            # Black points (color = 1)
            kernel(
                grid_size, block_size,
                (u, rhs, hx_inv2, hy_inv2, center_coeff, omega, nx, ny, 1)
            )
            cp.cuda.Device().synchronize()


class TransferKernels(CUDAKernels):
    """CUDA kernels for grid transfer operations."""
    
    def __init__(self, device_id: int = 0):
        """Initialize transfer kernels."""
        super().__init__(device_id)
        self._compile_transfer_kernels()
    
    def _compile_transfer_kernels(self) -> None:
        """Compile grid transfer kernels."""
        
        # Restriction kernel (full weighting)
        restriction_kernel_code = '''
        extern "C" __global__
        void restriction_kernel(
            const float* fine_grid,
            float* coarse_grid,
            const int fine_nx,
            const int fine_ny,
            const int coarse_nx,
            const int coarse_ny
        ) {
            int i_c = blockIdx.x * blockDim.x + threadIdx.x;
            int j_c = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i_c < coarse_nx && j_c < coarse_ny) {
                int i_f = 2 * i_c;
                int j_f = 2 * j_c;
                
                if (i_f < fine_nx && j_f < fine_ny) {
                    float sum = 0.0f;
                    float weight_sum = 0.0f;
                    
                    // Full weighting stencil
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            int ii = i_f + di;
                            int jj = j_f + dj;
                            
                            if (ii >= 0 && ii < fine_nx && jj >= 0 && jj < fine_ny) {
                                float weight;
                                if (di == 0 && dj == 0) weight = 4.0f;
                                else if (di == 0 || dj == 0) weight = 2.0f;
                                else weight = 1.0f;
                                
                                sum += weight * fine_grid[ii * fine_ny + jj];
                                weight_sum += weight;
                            }
                        }
                    }
                    
                    coarse_grid[i_c * coarse_ny + j_c] = sum / weight_sum;
                }
            }
        }
        '''
        
        # Prolongation kernel (bilinear interpolation)
        prolongation_kernel_code = '''
        extern "C" __global__
        void prolongation_kernel(
            const float* coarse_grid,
            float* fine_grid,
            const int fine_nx,
            const int fine_ny,
            const int coarse_nx,
            const int coarse_ny
        ) {
            int i_f = blockIdx.x * blockDim.x + threadIdx.x;
            int j_f = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i_f < fine_nx && j_f < fine_ny) {
                // Map fine grid point to coarse grid coordinates
                float i_c_real = 0.5f * i_f;
                float j_c_real = 0.5f * j_f;
                
                int i_c = (int)i_c_real;
                int j_c = (int)j_c_real;
                
                // Bilinear interpolation weights
                float wi = i_c_real - i_c;
                float wj = j_c_real - j_c;
                
                float value = 0.0f;
                
                // Bilinear interpolation from 4 coarse grid points
                if (i_c >= 0 && i_c < coarse_nx && j_c >= 0 && j_c < coarse_ny) {
                    value += (1.0f - wi) * (1.0f - wj) * coarse_grid[i_c * coarse_ny + j_c];
                }
                if (i_c + 1 < coarse_nx && j_c >= 0 && j_c < coarse_ny) {
                    value += wi * (1.0f - wj) * coarse_grid[(i_c + 1) * coarse_ny + j_c];
                }
                if (i_c >= 0 && i_c < coarse_nx && j_c + 1 < coarse_ny) {
                    value += (1.0f - wi) * wj * coarse_grid[i_c * coarse_ny + (j_c + 1)];
                }
                if (i_c + 1 < coarse_nx && j_c + 1 < coarse_ny) {
                    value += wi * wj * coarse_grid[(i_c + 1) * coarse_ny + (j_c + 1)];
                }
                
                fine_grid[i_f * fine_ny + j_f] = value;
            }
        }
        '''
        
        # Residual computation kernel
        residual_kernel_code = '''
        extern "C" __global__
        void residual_kernel(
            const float* u,
            const float* rhs,
            float* residual,
            const float hx_inv2,
            const float hy_inv2,
            const float center_coeff,
            const int nx,
            const int ny
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i < nx && j < ny) {
                int idx = i * ny + j;
                
                if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                    // Apply operator: Au
                    float Au = center_coeff * u[idx] +
                              hx_inv2 * (u[(i+1)*ny + j] + u[(i-1)*ny + j]) +
                              hy_inv2 * (u[i*ny + (j+1)] + u[i*ny + (j-1)]);
                    
                    // Compute residual: r = f - Au
                    residual[idx] = rhs[idx] - Au;
                } else {
                    // Boundary points
                    residual[idx] = 0.0f;
                }
            }
        }
        '''
        
        # Optimized restriction with shared memory
        optimized_restriction_code = '''
        extern "C" __global__
        void optimized_restriction_kernel(
            const float* fine_grid,
            float* coarse_grid,
            const int fine_nx,
            const int fine_ny,
            const int coarse_nx,
            const int coarse_ny
        ) {
            // Shared memory for fine grid block (18x18 includes halo)
            extern __shared__ float s_fine[];
            
            int i_c = blockIdx.x * blockDim.x + threadIdx.x;
            int j_c = blockIdx.y * blockDim.y + threadIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockDim.x;
            int by = blockDim.y;
            
            // Load fine grid data into shared memory with halo
            int i_f = 2 * i_c;
            int j_f = 2 * j_c;
            
            // Shared memory dimensions (block + 2 halo)
            int s_nx = bx + 2;
            int s_ny = by + 2;
            
            // Load main data
            if (i_f < fine_nx && j_f < fine_ny) {
                s_fine[(tx+1) * s_ny + (ty+1)] = fine_grid[i_f * fine_ny + j_f];
            }
            
            // Load halo data
            if (tx == 0 && i_f > 0) {
                s_fine[tx * s_ny + (ty+1)] = fine_grid[(i_f-1) * fine_ny + j_f];
            }
            if (tx == bx-1 && i_f+1 < fine_nx) {
                s_fine[(tx+2) * s_ny + (ty+1)] = fine_grid[(i_f+1) * fine_ny + j_f];
            }
            if (ty == 0 && j_f > 0) {
                s_fine[(tx+1) * s_ny + ty] = fine_grid[i_f * fine_ny + (j_f-1)];
            }
            if (ty == by-1 && j_f+1 < fine_ny) {
                s_fine[(tx+1) * s_ny + (ty+2)] = fine_grid[i_f * fine_ny + (j_f+1)];
            }
            
            // Load corner halo points
            if (tx == 0 && ty == 0 && i_f > 0 && j_f > 0) {
                s_fine[tx * s_ny + ty] = fine_grid[(i_f-1) * fine_ny + (j_f-1)];
            }
            if (tx == bx-1 && ty == 0 && i_f+1 < fine_nx && j_f > 0) {
                s_fine[(tx+2) * s_ny + ty] = fine_grid[(i_f+1) * fine_ny + (j_f-1)];
            }
            if (tx == 0 && ty == by-1 && i_f > 0 && j_f+1 < fine_ny) {
                s_fine[tx * s_ny + (ty+2)] = fine_grid[(i_f-1) * fine_ny + (j_f+1)];
            }
            if (tx == bx-1 && ty == by-1 && i_f+1 < fine_nx && j_f+1 < fine_ny) {
                s_fine[(tx+2) * s_ny + (ty+2)] = fine_grid[(i_f+1) * fine_ny + (j_f+1)];
            }
            
            __syncthreads();
            
            // Full-weighting restriction using shared memory
            if (i_c < coarse_nx && j_c < coarse_ny) {
                float sum = 0.0f;
                
                // 9-point stencil weights: center=4, face=2, corner=1, total=16
                sum += 4.0f * s_fine[(tx+1) * s_ny + (ty+1)];     // center
                sum += 2.0f * s_fine[tx * s_ny + (ty+1)];         // left
                sum += 2.0f * s_fine[(tx+2) * s_ny + (ty+1)];     // right  
                sum += 2.0f * s_fine[(tx+1) * s_ny + ty];         // bottom
                sum += 2.0f * s_fine[(tx+1) * s_ny + (ty+2)];     // top
                sum += 1.0f * s_fine[tx * s_ny + ty];             // bottom-left
                sum += 1.0f * s_fine[(tx+2) * s_ny + ty];         // bottom-right
                sum += 1.0f * s_fine[tx * s_ny + (ty+2)];         // top-left
                sum += 1.0f * s_fine[(tx+2) * s_ny + (ty+2)];     // top-right
                
                coarse_grid[i_c * coarse_ny + j_c] = sum / 16.0f;
            }
        }
        '''
        
        # Optimized prolongation with shared memory
        optimized_prolongation_code = '''
        extern "C" __global__
        void optimized_prolongation_kernel(
            const float* coarse_grid,
            float* fine_grid,
            const int fine_nx,
            const int fine_ny,
            const int coarse_nx,
            const int coarse_ny
        ) {
            // Shared memory for coarse grid block with halo
            extern __shared__ float s_coarse[];
            
            int i_f = blockIdx.x * blockDim.x + threadIdx.x;
            int j_f = blockIdx.y * blockDim.y + threadIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockDim.x;
            int by = blockDim.y;
            
            // Map to coarse grid
            int i_c = i_f / 2;
            int j_c = j_f / 2;
            
            // Shared memory coordinates
            int s_i = tx / 2 + 1;  // +1 for halo
            int s_j = ty / 2 + 1;
            int s_nx = bx / 2 + 3; // block/2 + 2 halo + 1
            
            // Load coarse grid data into shared memory
            if (tx % 2 == 0 && ty % 2 == 0 && i_c < coarse_nx && j_c < coarse_ny) {
                s_coarse[s_i * s_nx + s_j] = coarse_grid[i_c * coarse_ny + j_c];
                
                // Load halo
                if (tx == 0 && i_c > 0) {
                    s_coarse[(s_i-1) * s_nx + s_j] = coarse_grid[(i_c-1) * coarse_ny + j_c];
                }
                if (tx == bx-2 && i_c+1 < coarse_nx) {
                    s_coarse[(s_i+1) * s_nx + s_j] = coarse_grid[(i_c+1) * coarse_ny + j_c];
                }
                if (ty == 0 && j_c > 0) {
                    s_coarse[s_i * s_nx + (s_j-1)] = coarse_grid[i_c * coarse_ny + (j_c-1)];
                }
                if (ty == by-2 && j_c+1 < coarse_ny) {
                    s_coarse[s_i * s_nx + (s_j+1)] = coarse_grid[i_c * coarse_ny + (j_c+1)];
                }
            }
            
            __syncthreads();
            
            // Bilinear interpolation
            if (i_f < fine_nx && j_f < fine_ny) {
                float alpha = (i_f % 2) * 0.5f;
                float beta = (j_f % 2) * 0.5f;
                
                s_i = tx / 2 + 1;
                s_j = ty / 2 + 1;
                
                float val = (1-alpha)*(1-beta) * s_coarse[s_i * s_nx + s_j];
                if (s_i+1 < s_nx) val += alpha*(1-beta) * s_coarse[(s_i+1) * s_nx + s_j];
                if (s_j+1 < s_nx) val += (1-alpha)*beta * s_coarse[s_i * s_nx + (s_j+1)];
                if (s_i+1 < s_nx && s_j+1 < s_nx) val += alpha*beta * s_coarse[(s_i+1) * s_nx + (s_j+1)];
                
                fine_grid[i_f * fine_ny + j_f] = val;
            }
        }
        '''

        # Compile kernels
        self.compile_kernel("restriction_kernel", restriction_kernel_code)
        self.compile_kernel("prolongation_kernel", prolongation_kernel_code)
        self.compile_kernel("residual_kernel", residual_kernel_code)
        self.compile_kernel("optimized_restriction_kernel", optimized_restriction_code)
        self.compile_kernel("optimized_prolongation_kernel", optimized_prolongation_code)
    
    def restriction(
        self,
        fine_grid: 'cp.ndarray',
        coarse_grid: 'cp.ndarray'
    ) -> None:
        """
        Perform restriction operation (fine to coarse grid).
        
        Args:
            fine_grid: Input fine grid
            coarse_grid: Output coarse grid
        """
        fine_nx, fine_ny = fine_grid.shape
        coarse_nx, coarse_ny = coarse_grid.shape
        
        kernel = self.compiled_kernels["restriction_kernel"]
        
        # Configure kernel launch parameters
        block_size = self.get_optimal_block_size((coarse_nx, coarse_ny))
        grid_size = self.get_grid_size((coarse_nx, coarse_ny), block_size)
        
        kernel(
            grid_size, block_size,
            (fine_grid, coarse_grid, fine_nx, fine_ny, coarse_nx, coarse_ny)
        )
        
        cp.cuda.Device().synchronize()
    
    def prolongation(
        self,
        coarse_grid: 'cp.ndarray',
        fine_grid: 'cp.ndarray'
    ) -> None:
        """
        Perform prolongation operation (coarse to fine grid).
        
        Args:
            coarse_grid: Input coarse grid
            fine_grid: Output fine grid
        """
        fine_nx, fine_ny = fine_grid.shape
        coarse_nx, coarse_ny = coarse_grid.shape
        
        kernel = self.compiled_kernels["prolongation_kernel"]
        
        # Configure kernel launch parameters
        block_size = self.get_optimal_block_size((fine_nx, fine_ny))
        grid_size = self.get_grid_size((fine_nx, fine_ny), block_size)
        
        kernel(
            grid_size, block_size,
            (coarse_grid, fine_grid, fine_nx, fine_ny, coarse_nx, coarse_ny)
        )
        
        cp.cuda.Device().synchronize()
    
    def compute_residual(
        self,
        u: 'cp.ndarray',
        rhs: 'cp.ndarray',
        residual: 'cp.ndarray',
        hx: float,
        hy: float
    ) -> None:
        """
        Compute residual on GPU.
        
        Args:
            u: Solution array
            rhs: Right-hand side array
            residual: Output residual array
            hx: Grid spacing in x direction
            hy: Grid spacing in y direction
        """
        nx, ny = u.shape
        hx_inv2 = 1.0 / (hx * hx)
        hy_inv2 = 1.0 / (hy * hy)
        center_coeff = -2.0 * (hx_inv2 + hy_inv2)
        
        kernel = self.compiled_kernels["residual_kernel"]
        
        # Configure kernel launch parameters
        block_size = self.get_optimal_block_size((nx, ny))
        grid_size = self.get_grid_size((nx, ny), block_size)
        
        kernel(
            grid_size, block_size,
            (u, rhs, residual, hx_inv2, hy_inv2, center_coeff, nx, ny)
        )
        
        cp.cuda.Device().synchronize()


class MixedPrecisionKernels(CUDAKernels):
    """CUDA kernels for mixed-precision computations."""
    
    def __init__(self, device_id: int = 0):
        """Initialize mixed-precision kernels."""
        super().__init__(device_id)
        self._compile_mixed_precision_kernels()
    
    def _compile_mixed_precision_kernels(self) -> None:
        """Compile mixed-precision computation kernels."""
        
        # Mixed-precision residual kernel (single->double precision)
        mixed_residual_kernel_code = '''
        extern "C" __global__
        void mixed_precision_residual_kernel(
            const float* u,         // Solution in single precision
            const float* rhs,       // RHS in single precision  
            double* residual,       // Residual in double precision
            const double hx_inv2,   // Grid spacing parameters in double
            const double hy_inv2,
            const double center_coeff,
            const int nx,
            const int ny
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i < nx && j < ny) {
                int idx = i * ny + j;
                
                if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                    // Convert to double precision for computation
                    double u_center = (double)u[idx];
                    double u_left = (double)u[(i-1)*ny + j];
                    double u_right = (double)u[(i+1)*ny + j];
                    double u_bottom = (double)u[i*ny + (j-1)];
                    double u_top = (double)u[i*ny + (j+1)];
                    double rhs_val = (double)rhs[idx];
                    
                    // Compute operator Au with high precision
                    double Au = center_coeff * u_center +
                              hx_inv2 * (u_left + u_right) +
                              hy_inv2 * (u_bottom + u_top);
                    
                    // Compute residual: r = rhs - Au
                    residual[idx] = rhs_val - Au;
                } else {
                    // Boundary points
                    residual[idx] = 0.0;
                }
            }
        }
        '''
        
        # Precision conversion kernels
        float_to_double_kernel_code = '''
        extern "C" __global__
        void float_to_double_kernel(
            const float* input,
            double* output,
            const int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = (double)input[idx];
            }
        }
        '''
        
        double_to_float_kernel_code = '''
        extern "C" __global__
        void double_to_float_kernel(
            const double* input,
            float* output,
            const int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = (float)input[idx];
            }
        }
        '''
        
        # Mixed-precision correction kernel
        mixed_correction_kernel_code = '''
        extern "C" __global__
        void mixed_precision_correction_kernel(
            float* u_single,           // Solution in single precision (updated)
            const double* correction,  // Correction in double precision
            const float damping,       // Damping factor
            const int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                // Apply correction with damping: u_new = u_old + damping * correction
                u_single[idx] += damping * (float)correction[idx];
            }
        }
        '''
        
        # Compile mixed-precision kernels
        self.compile_kernel("mixed_precision_residual_kernel", mixed_residual_kernel_code)
        self.compile_kernel("float_to_double_kernel", float_to_double_kernel_code)
        self.compile_kernel("double_to_float_kernel", double_to_float_kernel_code)
        self.compile_kernel("mixed_correction_kernel", mixed_correction_kernel_code)
    
    def compute_mixed_precision_residual(
        self,
        u: 'cp.ndarray',      # Single precision
        rhs: 'cp.ndarray',    # Single precision
        hx: float,
        hy: float
    ) -> 'cp.ndarray':        # Double precision residual
        """
        Compute residual with mixed precision for enhanced accuracy.
        """
        nx, ny = u.shape
        hx_inv2 = 1.0 / (hx * hx)
        hy_inv2 = 1.0 / (hy * hy)
        center_coeff = -2.0 * (hx_inv2 + hy_inv2)
        
        # Allocate double precision residual
        residual = cp.zeros((nx, ny), dtype=cp.float64)
        
        kernel = self.compiled_kernels["mixed_precision_residual_kernel"]
        
        # Configure kernel launch parameters
        block_size = self.get_optimal_block_size((nx, ny))
        grid_size = self.get_grid_size((nx, ny), block_size)
        
        kernel(
            grid_size, block_size,
            (u, rhs, residual, hx_inv2, hy_inv2, center_coeff, nx, ny)
        )
        
        cp.cuda.Device().synchronize()
        return residual


class BlockStructuredKernels(CUDAKernels):
    """CUDA kernels for block-structured operations optimized for cache performance."""
    
    def __init__(self, device_id: int = 0):
        """Initialize block-structured kernels."""
        super().__init__(device_id)
        self._compile_block_kernels()
    
    def _compile_block_kernels(self) -> None:
        """Compile block-structured smoothing kernels."""
        
        # Block-structured Gauss-Seidel with cache optimization
        block_gauss_seidel_code = '''
        extern "C" __global__
        void block_gauss_seidel_kernel(
            float* u,
            const float* rhs,
            const float hx_inv2,
            const float hy_inv2,
            const float center_coeff,
            const int nx,
            const int ny,
            const int block_size,
            const int num_iterations
        ) {
            // Shared memory for block plus halo (block_size+2)^2
            extern __shared__ float s_u[];
            
            // Block and thread indices
            int block_i = blockIdx.x * block_size;
            int block_j = blockIdx.y * block_size;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            // Global indices
            int i = block_i + tx;
            int j = block_j + ty;
            
            // Shared memory size
            int s_size = block_size + 2;
            int s_idx = (tx + 1) * s_size + (ty + 1);
            
            for (int iter = 0; iter < num_iterations; iter++) {
                // Load block data into shared memory
                if (i < nx && j < ny) {
                    s_u[s_idx] = u[i * ny + j];
                    
                    // Load halo data
                    if (tx == 0 && i > 0) {
                        s_u[tx * s_size + (ty + 1)] = u[(i-1) * ny + j];
                    }
                    if (tx == block_size-1 && i < nx-1) {
                        s_u[(tx + 2) * s_size + (ty + 1)] = u[(i+1) * ny + j];
                    }
                    if (ty == 0 && j > 0) {
                        s_u[(tx + 1) * s_size + ty] = u[i * ny + (j-1)];
                    }
                    if (ty == block_size-1 && j < ny-1) {
                        s_u[(tx + 1) * s_size + (ty + 2)] = u[i * ny + (j+1)];
                    }
                }
                
                __syncthreads();
                
                // Block-structured Gauss-Seidel update
                if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
                    float neighbor_sum = hx_inv2 * (s_u[tx * s_size + (ty + 1)] + 
                                                   s_u[(tx + 2) * s_size + (ty + 1)]) +
                                       hy_inv2 * (s_u[(tx + 1) * s_size + ty] + 
                                                 s_u[(tx + 1) * s_size + (ty + 2)]);
                    
                    s_u[s_idx] = (rhs[i * ny + j] - neighbor_sum) / center_coeff;
                    u[i * ny + j] = s_u[s_idx];  // Write back immediately for G-S
                }
                
                __syncthreads();
            }
        }
        '''
        
        # Compile block kernels
        self.compile_kernel("block_gauss_seidel_kernel", block_gauss_seidel_code)
    
    def block_structured_smoothing(
        self,
        u: 'cp.ndarray',
        rhs: 'cp.ndarray',
        hx: float,
        hy: float,
        block_size: int = 16,
        num_iterations: int = 1
    ) -> None:
        """Perform block-structured smoothing optimized for cache performance."""
        nx, ny = u.shape
        hx_inv2 = 1.0 / (hx * hx)
        hy_inv2 = 1.0 / (hy * hy)
        center_coeff = -2.0 * (hx_inv2 + hy_inv2)
        
        kernel = self.compiled_kernels["block_gauss_seidel_kernel"]
        
        # Configure block-structured grid
        grid_blocks = ((nx + block_size - 1) // block_size,
                      (ny + block_size - 1) // block_size)
        thread_blocks = (block_size, block_size)
        
        # Shared memory size: (block_size + 2)^2 * sizeof(float)
        shared_mem_size = (block_size + 2) * (block_size + 2) * cp.float32().itemsize
        
        kernel(
            grid_blocks, thread_blocks,
            (u, rhs, hx_inv2, hy_inv2, center_coeff, nx, ny, block_size, num_iterations),
            shared_mem=shared_mem_size
        )
        
        cp.cuda.Device().synchronize()