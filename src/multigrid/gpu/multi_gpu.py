"""Multi-GPU support for distributed multigrid solvers."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..core.grid import Grid
from .memory_manager import GPUMemoryManager
from .gpu_solver import GPUMultigridSolver
from .gpu_precision import GPUPrecisionManager

logger = logging.getLogger(__name__)


@dataclass
class GPUDeviceInfo:
    """Information about GPU device capabilities."""
    device_id: int
    name: str
    total_memory_mb: float
    compute_capability: str
    multiprocessors: int
    max_threads_per_block: int
    available: bool = True
    current_load: float = 0.0


class MultiGPUManager:
    """
    Multi-GPU manager for distributed multigrid computations.
    
    Handles device allocation, memory management, and load balancing
    across multiple GPU devices.
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU manager.
        
        Args:
            device_ids: List of GPU device IDs to use. If None, uses all available.
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for multi-GPU support")
        
        # Discover available GPUs
        self.available_devices = self._discover_gpus()
        
        if device_ids is None:
            self.device_ids = list(range(len(self.available_devices)))
        else:
            self.device_ids = device_ids
        
        # Validate device IDs
        for device_id in self.device_ids:
            if device_id >= len(self.available_devices):
                raise ValueError(f"Device ID {device_id} not available")
        
        self.num_devices = len(self.device_ids)
        
        # Device management
        self.device_info: Dict[int, GPUDeviceInfo] = {}
        self.device_locks: Dict[int, threading.Lock] = {}
        self.memory_managers: Dict[int, GPUMemoryManager] = {}
        
        # Initialize devices
        self._initialize_devices()
        
        # Load balancing
        self.load_balancer = LoadBalancer(self.device_info)
        
        logger.info(f"Multi-GPU manager initialized: {self.num_devices} devices")
    
    def _discover_gpus(self) -> List[Dict[str, Any]]:
        """Discover available GPU devices."""
        devices = []
        
        try:
            num_devices = cp.cuda.runtime.getDeviceCount()
            
            for device_id in range(num_devices):
                with cp.cuda.Device(device_id):
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    meminfo = cp.cuda.runtime.memGetInfo()
                    
                    device_info = {
                        'device_id': device_id,
                        'name': props['name'].decode(),
                        'total_memory_mb': meminfo[1] / (1024 * 1024),
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'multiprocessors': props['multiProcessorCount'],
                        'max_threads_per_block': props['maxThreadsPerBlock']
                    }
                    
                    devices.append(device_info)
        
        except Exception as e:
            logger.error(f"Failed to discover GPU devices: {e}")
            raise
        
        logger.info(f"Discovered {len(devices)} GPU devices")
        return devices
    
    def _initialize_devices(self) -> None:
        """Initialize GPU devices and managers."""
        for device_id in self.device_ids:
            device_data = self.available_devices[device_id]
            
            self.device_info[device_id] = GPUDeviceInfo(
                device_id=device_id,
                name=device_data['name'],
                total_memory_mb=device_data['total_memory_mb'],
                compute_capability=device_data['compute_capability'],
                multiprocessors=device_data['multiprocessors'],
                max_threads_per_block=device_data['max_threads_per_block']
            )
            
            self.device_locks[device_id] = threading.Lock()
            
            # Initialize memory manager for each device
            try:
                self.memory_managers[device_id] = GPUMemoryManager(
                    device_id=device_id,
                    max_pool_size_mb=min(2048.0, device_data['total_memory_mb'] * 0.8)
                )
                logger.debug(f"Initialized memory manager for device {device_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize device {device_id}: {e}")
                self.device_info[device_id].available = False
    
    def get_optimal_device(self, memory_requirement_mb: float = 0) -> int:
        """
        Get optimal device for computation based on current load and requirements.
        
        Args:
            memory_requirement_mb: Required memory in MB
            
        Returns:
            Optimal device ID
        """
        return self.load_balancer.get_optimal_device(memory_requirement_mb)
    
    def allocate_device_for_task(
        self,
        task_name: str,
        memory_requirement_mb: float = 0,
        compute_requirement: float = 1.0
    ) -> Optional[int]:
        """
        Allocate device for specific task.
        
        Args:
            task_name: Name of the task
            memory_requirement_mb: Required memory
            compute_requirement: Relative compute requirement
            
        Returns:
            Allocated device ID or None if no device available
        """
        device_id = self.load_balancer.allocate_device(
            task_name, memory_requirement_mb, compute_requirement
        )
        
        if device_id is not None:
            logger.debug(f"Allocated device {device_id} for task {task_name}")
        else:
            logger.warning(f"No device available for task {task_name}")
        
        return device_id
    
    def release_device(self, device_id: int, task_name: str) -> None:
        """Release device from task."""
        self.load_balancer.release_device(device_id, task_name)
        logger.debug(f"Released device {device_id} from task {task_name}")
    
    def get_device_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all managed devices."""
        status = {}
        
        for device_id in self.device_ids:
            if device_id in self.device_info:
                info = self.device_info[device_id]
                memory_manager = self.memory_managers.get(device_id)
                
                device_status = {
                    'device_info': {
                        'name': info.name,
                        'total_memory_mb': info.total_memory_mb,
                        'compute_capability': info.compute_capability,
                        'available': info.available
                    },
                    'current_load': info.current_load,
                    'memory_usage': {}
                }
                
                if memory_manager:
                    device_status['memory_usage'] = memory_manager.get_memory_usage()
                
                status[device_id] = device_status
        
        return status
    
    def cleanup(self) -> None:
        """Clean up multi-GPU resources."""
        for memory_manager in self.memory_managers.values():
            try:
                memory_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up memory manager: {e}")
        
        self.memory_managers.clear()
        logger.info("Multi-GPU manager cleanup completed")


class LoadBalancer:
    """Load balancer for multi-GPU task allocation."""
    
    def __init__(self, device_info: Dict[int, GPUDeviceInfo]):
        """Initialize load balancer."""
        self.device_info = device_info
        self.device_loads: Dict[int, float] = {}
        self.active_tasks: Dict[int, List[str]] = {}
        self.lock = threading.RLock()
        
        # Initialize load tracking
        for device_id in device_info:
            self.device_loads[device_id] = 0.0
            self.active_tasks[device_id] = []
    
    def get_optimal_device(self, memory_requirement_mb: float = 0) -> int:
        """Get optimal device based on current load and requirements."""
        with self.lock:
            available_devices = [
                device_id for device_id, info in self.device_info.items()
                if info.available and info.total_memory_mb >= memory_requirement_mb
            ]
            
            if not available_devices:
                # Return least loaded device as fallback
                return min(self.device_loads.keys(), key=lambda d: self.device_loads[d])
            
            # Score devices based on load and memory availability
            device_scores = {}
            for device_id in available_devices:
                load_score = 1.0 - self.device_loads[device_id]  # Lower load is better
                memory_score = (self.device_info[device_id].total_memory_mb - memory_requirement_mb) / self.device_info[device_id].total_memory_mb
                
                # Weighted combination
                device_scores[device_id] = 0.6 * load_score + 0.4 * memory_score
            
            # Return device with highest score
            optimal_device = max(device_scores.keys(), key=lambda d: device_scores[d])
            return optimal_device
    
    def allocate_device(
        self,
        task_name: str,
        memory_requirement_mb: float = 0,
        compute_requirement: float = 1.0
    ) -> Optional[int]:
        """Allocate device for task."""
        with self.lock:
            device_id = self.get_optimal_device(memory_requirement_mb)
            
            if device_id in self.device_info and self.device_info[device_id].available:
                # Update load
                self.device_loads[device_id] = min(1.0, self.device_loads[device_id] + compute_requirement * 0.1)
                self.active_tasks[device_id].append(task_name)
                
                return device_id
            
            return None
    
    def release_device(self, device_id: int, task_name: str) -> None:
        """Release device from task."""
        with self.lock:
            if device_id in self.active_tasks and task_name in self.active_tasks[device_id]:
                self.active_tasks[device_id].remove(task_name)
                
                # Decay load over time
                self.device_loads[device_id] = max(0.0, self.device_loads[device_id] - 0.1)
    
    def update_device_load(self, device_id: int, load: float) -> None:
        """Update device load measurement."""
        with self.lock:
            if device_id in self.device_loads:
                self.device_loads[device_id] = max(0.0, min(1.0, load))


class DistributedMultigridSolver:
    """
    Distributed multigrid solver using multiple GPUs.
    
    Implements domain decomposition and parallel multigrid cycles
    across multiple GPU devices.
    """
    
    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        decomposition_strategy: str = "stripe",
        communication_method: str = "p2p",
        **solver_kwargs
    ):
        """
        Initialize distributed multigrid solver.
        
        Args:
            device_ids: List of GPU device IDs
            decomposition_strategy: Domain decomposition strategy
            communication_method: Inter-device communication method
            **solver_kwargs: Arguments for individual solvers
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for distributed multigrid solver")
        
        self.multi_gpu_manager = MultiGPUManager(device_ids)
        self.device_ids = self.multi_gpu_manager.device_ids
        self.num_devices = len(self.device_ids)
        
        self.decomposition_strategy = decomposition_strategy
        self.communication_method = communication_method
        self.solver_kwargs = solver_kwargs
        
        # Per-device solvers
        self.device_solvers: Dict[int, GPUMultigridSolver] = {}
        
        # Domain decomposition
        self.subdomain_info: Dict[int, Dict[str, Any]] = {}
        
        # Communication patterns
        self.communication_graph: Dict[int, List[int]] = {}
        
        logger.info(f"Distributed multigrid solver initialized: {self.num_devices} devices, "
                   f"strategy={decomposition_strategy}")
    
    def setup(self, global_grid: Grid, operator, restriction, prolongation) -> None:
        """
        Set up distributed multigrid solver.
        
        Args:
            global_grid: Global computational grid
            operator: Global operator
            restriction: Restriction operator
            prolongation: Prolongation operator
        """
        # Decompose domain
        self._decompose_domain(global_grid)
        
        # Setup communication graph
        self._setup_communication()
        
        # Initialize per-device solvers
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = []
            
            for device_id in self.device_ids:
                future = executor.submit(
                    self._setup_device_solver,
                    device_id, operator, restriction, prolongation
                )
                futures.append((device_id, future))
            
            # Wait for all setups to complete
            for device_id, future in futures:
                try:
                    future.result(timeout=60)  # 60 second timeout
                    logger.debug(f"Setup completed for device {device_id}")
                except Exception as e:
                    logger.error(f"Setup failed for device {device_id}: {e}")
                    raise
        
        logger.info("Distributed multigrid setup completed")
    
    def _decompose_domain(self, global_grid: Grid) -> None:
        """Decompose global domain across devices."""
        nx, ny = global_grid.shape
        
        if self.decomposition_strategy == "stripe":
            # Horizontal stripe decomposition
            rows_per_device = nx // self.num_devices
            
            for i, device_id in enumerate(self.device_ids):
                start_row = i * rows_per_device
                if i == self.num_devices - 1:
                    # Last device gets remaining rows
                    end_row = nx
                else:
                    end_row = (i + 1) * rows_per_device
                
                # Add overlap for boundary conditions
                if i > 0:
                    start_row -= 1  # Overlap with previous
                if i < self.num_devices - 1:
                    end_row += 1    # Overlap with next
                
                subdomain_nx = end_row - start_row
                subdomain_grid = Grid(
                    nx=subdomain_nx,
                    ny=ny,
                    domain=(
                        global_grid.domain[0] + start_row * global_grid.hx,
                        global_grid.domain[0] + end_row * global_grid.hx,
                        global_grid.domain[2],
                        global_grid.domain[3]
                    )
                )
                
                self.subdomain_info[device_id] = {
                    'grid': subdomain_grid,
                    'global_slice': (slice(start_row, end_row), slice(None)),
                    'interior_slice': (slice(1 if i > 0 else 0, -1 if i < self.num_devices - 1 else None), slice(None)),
                    'has_top_boundary': i == 0,
                    'has_bottom_boundary': i == self.num_devices - 1
                }
        
        elif self.decomposition_strategy == "checkerboard":
            # 2D checkerboard decomposition
            devices_per_dim = int(np.sqrt(self.num_devices))
            if devices_per_dim * devices_per_dim != self.num_devices:
                raise ValueError("Checkerboard decomposition requires square number of devices")
            
            rows_per_device = nx // devices_per_dim
            cols_per_device = ny // devices_per_dim
            
            for i, device_id in enumerate(self.device_ids):
                row_idx = i // devices_per_dim
                col_idx = i % devices_per_dim
                
                start_row = row_idx * rows_per_device
                end_row = (row_idx + 1) * rows_per_device if row_idx < devices_per_dim - 1 else nx
                start_col = col_idx * cols_per_device
                end_col = (col_idx + 1) * cols_per_device if col_idx < devices_per_dim - 1 else ny
                
                # Add overlap for boundaries
                if row_idx > 0:
                    start_row -= 1
                if row_idx < devices_per_dim - 1:
                    end_row += 1
                if col_idx > 0:
                    start_col -= 1
                if col_idx < devices_per_dim - 1:
                    end_col += 1
                
                subdomain_grid = Grid(
                    nx=end_row - start_row,
                    ny=end_col - start_col,
                    domain=(
                        global_grid.domain[0] + start_row * global_grid.hx,
                        global_grid.domain[0] + end_row * global_grid.hx,
                        global_grid.domain[2] + start_col * global_grid.hy,
                        global_grid.domain[2] + end_col * global_grid.hy
                    )
                )
                
                self.subdomain_info[device_id] = {
                    'grid': subdomain_grid,
                    'global_slice': (slice(start_row, end_row), slice(start_col, end_col)),
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'devices_per_dim': devices_per_dim
                }
        
        else:
            raise ValueError(f"Unknown decomposition strategy: {self.decomposition_strategy}")
    
    def _setup_communication(self) -> None:
        """Setup communication graph between devices."""
        for device_id in self.device_ids:
            self.communication_graph[device_id] = []
            
            if self.decomposition_strategy == "stripe":
                # Each device communicates with neighbors
                device_idx = self.device_ids.index(device_id)
                
                if device_idx > 0:
                    neighbor = self.device_ids[device_idx - 1]
                    self.communication_graph[device_id].append(neighbor)
                
                if device_idx < len(self.device_ids) - 1:
                    neighbor = self.device_ids[device_idx + 1]
                    self.communication_graph[device_id].append(neighbor)
            
            elif self.decomposition_strategy == "checkerboard":
                # Each device communicates with 4 neighbors
                subdomain = self.subdomain_info[device_id]
                row_idx = subdomain['row_idx']
                col_idx = subdomain['col_idx']
                devices_per_dim = subdomain['devices_per_dim']
                
                # Add neighbors (up, down, left, right)
                neighbors = []
                if row_idx > 0:
                    neighbors.append((row_idx - 1) * devices_per_dim + col_idx)
                if row_idx < devices_per_dim - 1:
                    neighbors.append((row_idx + 1) * devices_per_dim + col_idx)
                if col_idx > 0:
                    neighbors.append(row_idx * devices_per_dim + (col_idx - 1))
                if col_idx < devices_per_dim - 1:
                    neighbors.append(row_idx * devices_per_dim + (col_idx + 1))
                
                # Map to actual device IDs
                for neighbor_idx in neighbors:
                    if neighbor_idx < len(self.device_ids):
                        neighbor_device = self.device_ids[neighbor_idx]
                        self.communication_graph[device_id].append(neighbor_device)
    
    def _setup_device_solver(
        self,
        device_id: int,
        operator,
        restriction,
        prolongation
    ) -> None:
        """Setup solver for specific device."""
        subdomain = self.subdomain_info[device_id]
        
        # Create device-specific solver
        solver = GPUMultigridSolver(
            device_id=device_id,
            **self.solver_kwargs
        )
        
        # Setup solver with subdomain
        solver.setup(subdomain['grid'], operator, restriction, prolongation)
        
        self.device_solvers[device_id] = solver
    
    def solve(
        self,
        global_grid: Grid,
        operator,
        global_rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        precision_manager=None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve distributed multigrid problem.
        
        Args:
            global_grid: Global computational grid
            operator: Global operator
            global_rhs: Global right-hand side
            initial_guess: Initial solution guess
            precision_manager: Precision manager
            
        Returns:
            Tuple of (global_solution, convergence_info)
        """
        start_time = time.time()
        
        # Distribute RHS and initial guess
        local_rhs, local_initial = self._distribute_data(global_rhs, initial_guess)
        
        # Solve on each device in parallel
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = {}
            
            for device_id in self.device_ids:
                subdomain = self.subdomain_info[device_id]
                solver = self.device_solvers[device_id]
                
                future = executor.submit(
                    solver.solve,
                    subdomain['grid'],
                    operator,
                    local_rhs[device_id],
                    local_initial[device_id] if local_initial else None,
                    precision_manager
                )
                futures[device_id] = future
            
            # Collect results
            local_solutions = {}
            local_infos = {}
            
            for device_id, future in futures.items():
                try:
                    solution, info = future.result(timeout=300)  # 5 minute timeout
                    local_solutions[device_id] = solution
                    local_infos[device_id] = info
                except Exception as e:
                    logger.error(f"Solve failed on device {device_id}: {e}")
                    raise
        
        # Reconstruct global solution
        global_solution = self._reconstruct_global_solution(local_solutions, global_grid.shape)
        
        # Aggregate convergence info
        total_solve_time = time.time() - start_time
        convergence_info = self._aggregate_convergence_info(local_infos, total_solve_time)
        
        logger.info(f"Distributed solve completed: {convergence_info['total_iterations']} iterations, "
                   f"time={total_solve_time:.3f}s")
        
        return global_solution, convergence_info
    
    def _distribute_data(
        self,
        global_rhs: np.ndarray,
        global_initial: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
        """Distribute global data to subdomains."""
        local_rhs = {}
        local_initial = {} if global_initial is not None else None
        
        for device_id in self.device_ids:
            subdomain = self.subdomain_info[device_id]
            global_slice = subdomain['global_slice']
            
            local_rhs[device_id] = global_rhs[global_slice]
            
            if global_initial is not None:
                local_initial[device_id] = global_initial[global_slice]
        
        return local_rhs, local_initial
    
    def _reconstruct_global_solution(
        self,
        local_solutions: Dict[int, np.ndarray],
        global_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Reconstruct global solution from local solutions."""
        global_solution = np.zeros(global_shape)
        
        for device_id in self.device_ids:
            subdomain = self.subdomain_info[device_id]
            local_solution = local_solutions[device_id]
            
            if self.decomposition_strategy == "stripe":
                # Extract interior part (remove overlap)
                interior_slice = subdomain['interior_slice']
                global_slice = subdomain['global_slice']
                
                # Map local interior to global coordinates
                local_interior = local_solution[interior_slice]
                
                # Calculate global slice for interior
                global_start_row = global_slice[0].start
                if not subdomain['has_top_boundary']:
                    global_start_row += 1  # Skip overlap
                
                global_end_row = global_slice[0].stop
                if not subdomain['has_bottom_boundary']:
                    global_end_row -= 1  # Skip overlap
                
                global_interior_slice = (slice(global_start_row, global_end_row), slice(None))
                global_solution[global_interior_slice] = local_interior
            
            else:  # checkerboard
                # Similar logic for 2D decomposition
                global_slice = subdomain['global_slice']
                # For simplicity, assume no overlap handling for now
                global_solution[global_slice] = local_solution
        
        return global_solution
    
    def _aggregate_convergence_info(
        self,
        local_infos: Dict[int, Dict[str, Any]],
        total_time: float
    ) -> Dict[str, Any]:
        """Aggregate convergence information from all devices."""
        if not local_infos:
            return {'error': 'No convergence info available'}
        
        # Take statistics from all devices
        iterations = [info['iterations'] for info in local_infos.values()]
        final_residuals = [info['final_residual'] for info in local_infos.values()]
        converged_flags = [info['converged'] for info in local_infos.values()]
        
        aggregated_info = {
            'total_iterations': max(iterations),
            'average_iterations': np.mean(iterations),
            'final_residual': max(final_residuals),  # Worst case
            'converged': all(converged_flags),
            'num_devices': len(local_infos),
            'distributed_solve_time': total_time,
            'decomposition_strategy': self.decomposition_strategy,
            'device_stats': {
                device_id: {
                    'iterations': info['iterations'],
                    'final_residual': info['final_residual'],
                    'converged': info['converged'],
                    'solve_time': info.get('gpu_solve_time', 0)
                }
                for device_id, info in local_infos.items()
            }
        }
        
        return aggregated_info
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics from all devices."""
        device_stats = {}
        
        for device_id, solver in self.device_solvers.items():
            try:
                stats = solver.get_performance_statistics()
                device_stats[device_id] = stats
            except Exception as e:
                logger.warning(f"Failed to get stats from device {device_id}: {e}")
                device_stats[device_id] = {'error': str(e)}
        
        return {
            'multi_gpu_stats': {
                'num_devices': self.num_devices,
                'decomposition_strategy': self.decomposition_strategy,
                'communication_method': self.communication_method
            },
            'device_statistics': device_stats,
            'load_balancer_stats': {
                'device_loads': self.multi_gpu_manager.load_balancer.device_loads,
                'active_tasks': {
                    device_id: len(tasks) 
                    for device_id, tasks in self.multi_gpu_manager.load_balancer.active_tasks.items()
                }
            }
        }
    
    def cleanup(self) -> None:
        """Clean up distributed solver resources."""
        # Cleanup device solvers
        for solver in self.device_solvers.values():
            try:
                solver.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up device solver: {e}")
        
        # Cleanup multi-GPU manager
        self.multi_gpu_manager.cleanup()
        
        logger.info("Distributed multigrid solver cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass