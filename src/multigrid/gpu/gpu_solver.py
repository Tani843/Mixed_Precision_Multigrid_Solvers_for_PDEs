"""GPU-accelerated multigrid solver implementations."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..core.grid import Grid
from ..core.precision import PrecisionManager
from .memory_manager import GPUMemoryManager
from .cuda_kernels import SmoothingKernels, TransferKernels
from .gpu_precision import GPUPrecisionManager, GPUPrecisionLevel

logger = logging.getLogger(__name__)


class GPUMultigridSolver:
    """
    GPU-accelerated multigrid solver with CUDA optimization.
    
    Implements V-cycle, W-cycle, and F-cycle multigrid methods
    with GPU acceleration and mixed-precision support.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        max_levels: int = 6,
        cycle_type: str = "V",
        pre_smooth_iterations: int = 2,
        post_smooth_iterations: int = 2,
        coarse_solver_iterations: int = 10,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        smoother: str = "jacobi",
        enable_mixed_precision: bool = True,
        use_tensor_cores: bool = True,
        memory_pool_size_mb: float = 2048.0
    ):
        """
        Initialize GPU multigrid solver.
        
        Args:
            device_id: GPU device ID
            max_levels: Maximum number of multigrid levels
            cycle_type: Cycle type ("V", "W", "F")
            pre_smooth_iterations: Pre-smoothing iterations
            post_smooth_iterations: Post-smoothing iterations
            coarse_solver_iterations: Coarse grid solver iterations
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
            smoother: Smoothing method ("jacobi", "gauss_seidel", "sor")
            enable_mixed_precision: Enable mixed precision
            use_tensor_cores: Enable Tensor Core optimization
            memory_pool_size_mb: GPU memory pool size in MB
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU multigrid solver")
        
        self.device_id = device_id
        self.max_levels = max_levels
        self.cycle_type = cycle_type.upper()
        self.pre_smooth_iterations = pre_smooth_iterations
        self.post_smooth_iterations = post_smooth_iterations
        self.coarse_solver_iterations = coarse_solver_iterations
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.smoother = smoother
        
        # Initialize GPU components
        with cp.cuda.Device(device_id):
            self.memory_manager = GPUMemoryManager(
                device_id=device_id,
                max_pool_size_mb=memory_pool_size_mb
            )
            
            self.smoothing_kernels = SmoothingKernels(device_id)
            self.transfer_kernels = TransferKernels(device_id)
            
            if enable_mixed_precision:
                self.precision_manager = GPUPrecisionManager(
                    device_id=device_id,
                    enable_tensor_cores=use_tensor_cores,
                    adaptive=True
                )
            else:
                self.precision_manager = None
        
        # Grid hierarchy (will be set up during solve)
        self.grids: List[Grid] = []
        self.gpu_arrays: Dict[str, List['cp.ndarray']] = {}
        
        # Performance tracking
        self.solve_stats = {
            'total_iterations': 0,
            'total_solve_time': 0.0,
            'gpu_transfer_time': 0.0,
            'kernel_execution_time': 0.0,
            'memory_usage_mb': 0.0,
            'convergence_history': []
        }
        
        self.name = f"GPU-MG-{cycle_type}cycle"
        
        logger.info(f"GPU multigrid solver initialized: device={device_id}, "
                   f"cycles={cycle_type}, smoother={smoother}, "
                   f"mixed_precision={enable_mixed_precision}")
    
    def setup(self, fine_grid: Grid, operator, restriction, prolongation) -> None:
        """
        Set up multigrid hierarchy on GPU.
        
        Args:
            fine_grid: Finest grid level
            operator: Discrete operator
            restriction: Restriction operator
            prolongation: Prolongation operator
        """
        with cp.cuda.Device(self.device_id):
            # Build grid hierarchy
            self.grids = [fine_grid]
            current_grid = fine_grid
            
            for level in range(1, self.max_levels):
                try:
                    coarse_grid = current_grid.coarsen()
                    self.grids.append(coarse_grid)
                    current_grid = coarse_grid
                    
                    # Stop if grid becomes too coarse
                    if min(coarse_grid.nx, coarse_grid.ny) <= 3:
                        break
                except ValueError:
                    break
            
            # Allocate GPU arrays for each level
            self._allocate_gpu_arrays()
            
            # Store operators
            self.operator = operator
            self.restriction = restriction
            self.prolongation = prolongation
            
            logger.info(f"GPU multigrid hierarchy set up: {len(self.grids)} levels, "
                       f"finest={self.grids[0].shape}, coarsest={self.grids[-1].shape}")
    
    def _allocate_gpu_arrays(self) -> None:
        """Allocate GPU arrays for all grid levels."""
        self.gpu_arrays = {
            'u': [],
            'rhs': [],
            'residual': [],
            'correction': []
        }
        
        for level, grid in enumerate(self.grids):
            # Get optimal precision for each level
            if self.precision_manager:
                dtype = self.precision_manager.get_optimal_dtype('smoothing', level)
            else:
                dtype = np.float32
            
            # Allocate arrays
            self.gpu_arrays['u'].append(
                self.memory_manager.allocate_gpu_array(grid.shape, dtype)
            )
            self.gpu_arrays['rhs'].append(
                self.memory_manager.allocate_gpu_array(grid.shape, dtype)
            )
            self.gpu_arrays['residual'].append(
                self.memory_manager.allocate_gpu_array(grid.shape, dtype)
            )
            self.gpu_arrays['correction'].append(
                self.memory_manager.allocate_gpu_array(grid.shape, dtype)
            )
        
        logger.debug(f"Allocated GPU arrays for {len(self.grids)} levels")
    
    def solve(
        self,
        grid: Grid,
        operator,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional[PrecisionManager] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve linear system using GPU-accelerated multigrid.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            rhs: Right-hand side
            initial_guess: Initial solution guess
            precision_manager: CPU precision manager (for compatibility)
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        solve_start_time = time.time()
        
        with cp.cuda.Device(self.device_id):
            # Transfer data to GPU
            transfer_start = time.time()
            
            gpu_rhs = self.memory_manager.to_gpu(rhs)
            
            if initial_guess is not None:
                gpu_u = self.memory_manager.to_gpu(initial_guess)
            else:
                gpu_u = self.memory_manager.allocate_like_gpu(rhs, zero_fill=True)
            
            self.solve_stats['gpu_transfer_time'] += time.time() - transfer_start
            
            # Apply precision conversion if needed
            if self.precision_manager:
                gpu_u = self.precision_manager.convert_to_optimal_precision(
                    gpu_u, 'smoothing', 0
                )
                gpu_rhs = self.precision_manager.convert_to_optimal_precision(
                    gpu_rhs, 'smoothing', 0
                )
            
            # Copy to level 0 arrays
            cp.copyto(self.gpu_arrays['u'][0], gpu_u)
            cp.copyto(self.gpu_arrays['rhs'][0], gpu_rhs)
            
            # Initial residual
            self.transfer_kernels.compute_residual(
                self.gpu_arrays['u'][0],
                self.gpu_arrays['rhs'][0],
                self.gpu_arrays['residual'][0],
                grid.hx, grid.hy
            )
            
            initial_residual = float(cp.linalg.norm(self.gpu_arrays['residual'][0]))
            current_residual = initial_residual
            
            convergence_info = {
                'initial_residual': initial_residual,
                'iterations': 0,
                'converged': False,
                'residual_history': [initial_residual]
            }
            
            # Main multigrid iteration loop
            for iteration in range(self.max_iterations):
                iteration_start = time.time()
                
                # Perform multigrid cycle
                self._multigrid_cycle(0)
                
                # Compute new residual
                self.transfer_kernels.compute_residual(
                    self.gpu_arrays['u'][0],
                    self.gpu_arrays['rhs'][0],
                    self.gpu_arrays['residual'][0],
                    grid.hx, grid.hy
                )
                
                current_residual = float(cp.linalg.norm(self.gpu_arrays['residual'][0]))
                convergence_info['residual_history'].append(current_residual)
                
                # Update adaptive precision if enabled
                if self.precision_manager and self.precision_manager.adaptive:
                    memory_usage = self.memory_manager.get_memory_usage()
                    gpu_memory_mb = memory_usage['gpu_memory_info']['used_gpu_memory'] / (1024**2)
                    
                    self.precision_manager.update_precision_adaptive(
                        current_residual,
                        [grid.shape for grid in self.grids],
                        gpu_memory_mb
                    )
                
                iteration_time = time.time() - iteration_start
                self.solve_stats['kernel_execution_time'] += iteration_time
                
                # Check convergence
                if current_residual < self.tolerance:
                    convergence_info['converged'] = True
                    break
                
                # Store convergence data
                self.solve_stats['convergence_history'].append({
                    'iteration': iteration,
                    'residual': current_residual,
                    'time': iteration_time
                })
            
            convergence_info['iterations'] = iteration + 1
            convergence_info['final_residual'] = current_residual
            
            # Transfer solution back to CPU
            transfer_start = time.time()
            solution = self.memory_manager.to_cpu(self.gpu_arrays['u'][0])
            self.solve_stats['gpu_transfer_time'] += time.time() - transfer_start
            
            # Update solve statistics
            total_solve_time = time.time() - solve_start_time
            self.solve_stats['total_solve_time'] += total_solve_time
            self.solve_stats['total_iterations'] += convergence_info['iterations']
            
            # Add GPU-specific info to convergence info
            convergence_info.update({
                'device_id': self.device_id,
                'cycle_type': self.cycle_type,
                'num_levels': len(self.grids),
                'smoother': self.smoother,
                'gpu_solve_time': total_solve_time,
                'gpu_transfer_time': self.solve_stats['gpu_transfer_time'],
                'kernel_time': self.solve_stats['kernel_execution_time']
            })
            
            # Add precision statistics if available
            if self.precision_manager:
                convergence_info['precision_stats'] = self.precision_manager.get_precision_statistics()
            
            logger.info(f"GPU solve completed: {convergence_info['iterations']} iterations, "
                       f"residual={current_residual:.2e}, time={total_solve_time:.3f}s")
            
            return solution, convergence_info
    
    def _multigrid_cycle(self, level: int) -> None:
        """
        Perform one multigrid cycle recursively.
        
        Args:
            level: Current grid level
        """
        if level == len(self.grids) - 1:
            # Coarsest level - solve directly
            self._solve_coarsest_level(level)
            return
        
        current_grid = self.grids[level]
        
        # Pre-smoothing
        if self.pre_smooth_iterations > 0:
            self._smooth(level, self.pre_smooth_iterations)
        
        # Compute residual
        self.transfer_kernels.compute_residual(
            self.gpu_arrays['u'][level],
            self.gpu_arrays['rhs'][level],
            self.gpu_arrays['residual'][level],
            current_grid.hx, current_grid.hy
        )
        
        # Restrict residual to coarser level
        self.transfer_kernels.restriction(
            self.gpu_arrays['residual'][level],
            self.gpu_arrays['rhs'][level + 1]
        )
        
        # Zero initial guess for correction
        self.gpu_arrays['u'][level + 1].fill(0)
        
        # Recursive call(s) based on cycle type
        if self.cycle_type == "V":
            self._multigrid_cycle(level + 1)
        elif self.cycle_type == "W":
            self._multigrid_cycle(level + 1)
            self._multigrid_cycle(level + 1)  # Second call for W-cycle
        elif self.cycle_type == "F":
            # F-cycle: V-cycle followed by F-cycle
            if level == 0:
                self._multigrid_cycle(level + 1)  # V-cycle
            self._multigrid_cycle(level + 1)  # F-cycle recursion
        
        # Prolongate correction and add to current solution
        self.transfer_kernels.prolongation(
            self.gpu_arrays['u'][level + 1],
            self.gpu_arrays['correction'][level]
        )
        
        # Add correction with Tensor Core optimization if available
        if self.precision_manager and self.precision_manager.tensor_core_available:
            self.gpu_arrays['u'][level] = self.precision_manager.apply_tensor_core_optimization(
                self.gpu_arrays['u'][level],
                self.gpu_arrays['correction'][level],
                "add"
            )
        else:
            self.gpu_arrays['u'][level] += self.gpu_arrays['correction'][level]
        
        # Post-smoothing
        if self.post_smooth_iterations > 0:
            self._smooth(level, self.post_smooth_iterations)
    
    def _smooth(self, level: int, iterations: int) -> None:
        """
        Apply smoothing at given level.
        
        Args:
            level: Grid level
            iterations: Number of smoothing iterations
        """
        grid = self.grids[level]
        
        if self.smoother == "jacobi":
            # Need temporary array for Jacobi
            temp_u = self.memory_manager.allocate_like_gpu(self.gpu_arrays['u'][level])
            
            self.smoothing_kernels.jacobi_smoothing(
                self.gpu_arrays['u'][level],
                temp_u,
                self.gpu_arrays['rhs'][level],
                grid.hx, grid.hy,
                iterations,
                use_shared_memory=True
            )
            
            cp.copyto(self.gpu_arrays['u'][level], temp_u)
            self.memory_manager.memory_pool.deallocate(temp_u)
            
        elif self.smoother == "gauss_seidel":
            self.smoothing_kernels.red_black_gauss_seidel(
                self.gpu_arrays['u'][level],
                self.gpu_arrays['rhs'][level],
                grid.hx, grid.hy,
                iterations
            )
            
        elif self.smoother == "sor":
            # Optimal omega for 2D Poisson equation
            omega = 2.0 / (1.0 + np.sin(np.pi / max(grid.nx, grid.ny)))
            
            self.smoothing_kernels.sor_smoothing(
                self.gpu_arrays['u'][level],
                self.gpu_arrays['rhs'][level],
                grid.hx, grid.hy,
                omega,
                iterations
            )
    
    def _solve_coarsest_level(self, level: int) -> None:
        """Solve on coarsest level using iterative method."""
        # Use many iterations of smoother on coarsest level
        self._smooth(level, self.coarse_solver_iterations)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'solver_stats': self.solve_stats.copy(),
            'memory_usage': self.memory_manager.get_memory_usage(),
            'gpu_info': {
                'device_id': self.device_id,
                'cycle_type': self.cycle_type,
                'num_levels': len(self.grids),
                'smoother': self.smoother
            }
        }
        
        if self.precision_manager:
            stats['precision_stats'] = self.precision_manager.get_precision_statistics()
        
        # Calculate performance metrics
        if self.solve_stats['total_solve_time'] > 0:
            stats['performance_metrics'] = {
                'average_iteration_time': (
                    self.solve_stats['kernel_execution_time'] / 
                    max(self.solve_stats['total_iterations'], 1)
                ),
                'gpu_utilization': (
                    self.solve_stats['kernel_execution_time'] / 
                    self.solve_stats['total_solve_time']
                ) * 100,
                'transfer_overhead': (
                    self.solve_stats['gpu_transfer_time'] / 
                    self.solve_stats['total_solve_time']
                ) * 100
            }
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up GPU resources."""
        try:
            # Clear GPU arrays
            self.gpu_arrays.clear()
            
            # Cleanup memory manager
            self.memory_manager.cleanup()
            
            logger.debug("GPU multigrid solver cleanup completed")
        except Exception as e:
            logger.warning(f"Error during GPU cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


class GPUCommunicationAvoidingMultigrid(GPUMultigridSolver):
    """
    GPU-accelerated Communication-Avoiding Multigrid solver.
    
    Extends the base GPU solver with communication-avoiding optimizations,
    block-structured operations, and advanced memory management.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        max_levels: int = 6,
        cycle_type: str = "V",
        block_size: int = 32,
        enable_memory_pool: bool = True,
        use_fmg: bool = False,
        fmg_cycles: int = 1,
        async_operations: bool = True,
        **kwargs
    ):
        """
        Initialize GPU Communication-Avoiding Multigrid solver.
        
        Args:
            device_id: GPU device ID
            max_levels: Maximum multigrid levels
            cycle_type: Multigrid cycle type
            block_size: Block size for communication-avoiding operations
            enable_memory_pool: Enable memory pool optimization
            use_fmg: Use Full Multigrid initialization
            fmg_cycles: Number of FMG cycles
            async_operations: Enable asynchronous GPU operations
            **kwargs: Additional arguments for base solver
        """
        super().__init__(device_id=device_id, max_levels=max_levels, 
                        cycle_type=cycle_type, **kwargs)
        
        self.block_size = block_size
        self.enable_memory_pool = enable_memory_pool
        self.use_fmg = use_fmg
        self.fmg_cycles = fmg_cycles
        self.async_operations = async_operations
        
        # Advanced performance tracking
        self.ca_stats = {
            'block_operations': 0,
            'async_operations': 0,
            'fmg_initializations': 0,
            'memory_pool_hits': 0
        }
        
        self.name = f"GPU-CA-MG-{cycle_type}cycle"
        
        logger.info(f"GPU CA-Multigrid solver initialized: block_size={block_size}, "
                   f"fmg={use_fmg}, async={async_operations}")
    
    def solve(
        self,
        grid: Grid,
        operator,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager: Optional[PrecisionManager] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve with Communication-Avoiding optimizations.
        
        Args:
            grid: Computational grid
            operator: Linear operator
            rhs: Right-hand side
            initial_guess: Initial solution guess
            precision_manager: CPU precision manager
            
        Returns:
            Tuple of (solution, convergence_info)
        """
        with cp.cuda.Device(self.device_id):
            # Full Multigrid initialization if enabled
            if self.use_fmg and initial_guess is None:
                initial_guess = self._full_multigrid_initialization(grid, operator, rhs)
                self.ca_stats['fmg_initializations'] += 1
            
            # Call parent solve method with optimized initial guess
            solution, info = super().solve(
                grid, operator, rhs, initial_guess, precision_manager
            )
            
            # Add CA-specific statistics
            info.update({
                'ca_optimizations': True,
                'block_size': self.block_size,
                'fmg_used': self.use_fmg,
                'async_operations': self.async_operations,
                'ca_stats': self.ca_stats.copy()
            })
            
            return solution, info
    
    def _full_multigrid_initialization(
        self,
        grid: Grid,
        operator,
        rhs: np.ndarray
    ) -> np.ndarray:
        """
        Perform Full Multigrid initialization.
        
        Args:
            grid: Fine grid
            operator: Linear operator
            rhs: Right-hand side
            
        Returns:
            Initial guess from FMG
        """
        # Start from coarsest level and work up
        current_rhs = self.memory_manager.to_gpu(rhs)
        
        # Restrict RHS to all levels
        for level in range(len(self.grids) - 1):
            coarse_rhs = self.memory_manager.allocate_gpu_array(
                self.grids[level + 1].shape, current_rhs.dtype
            )
            
            self.transfer_kernels.restriction(current_rhs, coarse_rhs)
            current_rhs = coarse_rhs
        
        # Solve on coarsest level
        coarsest_level = len(self.grids) - 1
        cp.copyto(self.gpu_arrays['rhs'][coarsest_level], current_rhs)
        self.gpu_arrays['u'][coarsest_level].fill(0)
        
        self._solve_coarsest_level(coarsest_level)
        
        # Prolongate solution to finer levels
        for level in range(coarsest_level - 1, -1, -1):
            # Prolongate
            self.transfer_kernels.prolongation(
                self.gpu_arrays['u'][level + 1],
                self.gpu_arrays['u'][level]
            )
            
            # Perform FMG cycles at this level
            for _ in range(self.fmg_cycles):
                self._smooth(level, self.pre_smooth_iterations)
        
        # Return CPU solution
        return self.memory_manager.to_cpu(self.gpu_arrays['u'][0])
    
    def _multigrid_cycle(self, level: int) -> None:
        """
        Communication-avoiding multigrid cycle with optimizations.
        
        Args:
            level: Current grid level
        """
        if level == len(self.grids) - 1:
            self._solve_coarsest_level(level)
            return
        
        current_grid = self.grids[level]
        
        # Use asynchronous operations if enabled
        if self.async_operations:
            stream = self.memory_manager.get_next_stream()
            with stream:
                self._ca_cycle_async(level)
                self.ca_stats['async_operations'] += 1
        else:
            self._ca_cycle_sync(level)
    
    def _ca_cycle_async(self, level: int) -> None:
        """Asynchronous communication-avoiding cycle."""
        # Pre-smoothing with block operations
        if self.pre_smooth_iterations > 0:
            self._block_smooth(level, self.pre_smooth_iterations)
        
        # Compute and restrict residual
        self._block_restrict_residual(level)
        
        # Recursive calls
        if self.cycle_type == "V":
            self._multigrid_cycle(level + 1)
        elif self.cycle_type == "W":
            self._multigrid_cycle(level + 1)
            self._multigrid_cycle(level + 1)
        
        # Prolongate and correct
        self._block_prolongate_correct(level)
        
        # Post-smoothing
        if self.post_smooth_iterations > 0:
            self._block_smooth(level, self.post_smooth_iterations)
    
    def _ca_cycle_sync(self, level: int) -> None:
        """Synchronous communication-avoiding cycle."""
        # Same as async but without stream context
        self._ca_cycle_async(level)
    
    def _block_smooth(self, level: int, iterations: int) -> None:
        """
        Block-structured smoothing operation.
        
        Args:
            level: Grid level
            iterations: Number of iterations
        """
        # Use shared memory optimization for block operations
        if self.smoother == "jacobi":
            temp_u = self.memory_manager.allocate_like_gpu(self.gpu_arrays['u'][level])
            
            self.smoothing_kernels.jacobi_smoothing(
                self.gpu_arrays['u'][level],
                temp_u,
                self.gpu_arrays['rhs'][level],
                self.grids[level].hx,
                self.grids[level].hy,
                iterations,
                use_shared_memory=True  # Enable shared memory optimization
            )
            
            cp.copyto(self.gpu_arrays['u'][level], temp_u)
            self.memory_manager.memory_pool.deallocate(temp_u)
            
            self.ca_stats['block_operations'] += 1
        else:
            # Fall back to standard smoothing
            self._smooth(level, iterations)
    
    def _block_restrict_residual(self, level: int) -> None:
        """Block-optimized residual computation and restriction."""
        grid = self.grids[level]
        
        # Compute residual
        self.transfer_kernels.compute_residual(
            self.gpu_arrays['u'][level],
            self.gpu_arrays['rhs'][level],
            self.gpu_arrays['residual'][level],
            grid.hx, grid.hy
        )
        
        # Restrict to coarser level
        self.transfer_kernels.restriction(
            self.gpu_arrays['residual'][level],
            self.gpu_arrays['rhs'][level + 1]
        )
        
        self.ca_stats['block_operations'] += 1
    
    def _block_prolongate_correct(self, level: int) -> None:
        """Block-optimized prolongation and correction."""
        # Prolongate correction
        self.transfer_kernels.prolongation(
            self.gpu_arrays['u'][level + 1],
            self.gpu_arrays['correction'][level]
        )
        
        # Add correction with Tensor Core optimization
        if self.precision_manager and self.precision_manager.tensor_core_available:
            self.gpu_arrays['u'][level] = self.precision_manager.apply_tensor_core_optimization(
                self.gpu_arrays['u'][level],
                self.gpu_arrays['correction'][level],
                "add"
            )
        else:
            self.gpu_arrays['u'][level] += self.gpu_arrays['correction'][level]
        
        self.ca_stats['block_operations'] += 1
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get CA-specific performance statistics."""
        base_stats = super().get_performance_statistics()
        
        # Add CA-specific metrics
        base_stats['ca_optimizations'] = {
            'block_size': self.block_size,
            'ca_stats': self.ca_stats.copy(),
            'memory_pool_enabled': self.enable_memory_pool,
            'fmg_enabled': self.use_fmg,
            'async_enabled': self.async_operations
        }
        
        # Calculate CA-specific performance metrics
        total_ops = sum(self.ca_stats.values())
        if total_ops > 0:
            base_stats['ca_performance_metrics'] = {
                'block_operation_ratio': self.ca_stats['block_operations'] / total_ops,
                'async_operation_ratio': self.ca_stats['async_operations'] / total_ops,
                'memory_pool_hit_rate': (
                    self.memory_manager.memory_pool.get_statistics()['hit_rate'] * 100
                    if self.enable_memory_pool else 0
                )
            }
        
        return base_stats