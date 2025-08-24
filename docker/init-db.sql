-- Database initialization script for multigrid results storage
-- This script creates tables for storing benchmark results, performance metrics, and analysis data

-- Create database schema for multigrid results
CREATE SCHEMA IF NOT EXISTS multigrid;

-- Table for storing benchmark results
CREATE TABLE IF NOT EXISTS multigrid.benchmark_results (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    benchmark_name VARCHAR(255) NOT NULL,
    problem_size VARCHAR(50) NOT NULL,
    grid_dimensions VARCHAR(50) NOT NULL,
    solver_type VARCHAR(100) NOT NULL,
    precision_mode VARCHAR(50) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    execution_time_seconds DECIMAL(12, 6) NOT NULL,
    memory_usage_mb INTEGER,
    iterations_count INTEGER,
    final_residual DECIMAL(20, 15),
    convergence_rate DECIMAL(8, 6),
    energy_consumption_joules DECIMAL(12, 3),
    gpu_utilization_percent DECIMAL(5, 2),
    cpu_utilization_percent DECIMAL(5, 2),
    metadata JSONB
);

-- Table for storing performance profiles
CREATE TABLE IF NOT EXISTS multigrid.performance_profiles (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    profile_name VARCHAR(255) NOT NULL,
    solver_config JSONB NOT NULL,
    performance_data JSONB NOT NULL,
    system_info JSONB NOT NULL,
    tags TEXT[]
);

-- Table for storing validation results
CREATE TABLE IF NOT EXISTS multigrid.validation_results (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    test_name VARCHAR(255) NOT NULL,
    test_category VARCHAR(100) NOT NULL,
    problem_description TEXT,
    grid_size VARCHAR(50) NOT NULL,
    analytical_solution TEXT,
    numerical_solution_hash VARCHAR(64),
    l2_error DECIMAL(20, 15) NOT NULL,
    max_error DECIMAL(20, 15) NOT NULL,
    convergence_order DECIMAL(8, 4),
    validation_status VARCHAR(50) NOT NULL,
    error_analysis JSONB,
    statistical_data JSONB
);

-- Table for storing system metrics over time
CREATE TABLE IF NOT EXISTS multigrid.system_metrics (
    id SERIAL PRIMARY KEY,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    hostname VARCHAR(255) NOT NULL,
    cpu_usage_percent DECIMAL(5, 2),
    memory_usage_percent DECIMAL(5, 2),
    memory_available_mb BIGINT,
    gpu_count INTEGER DEFAULT 0,
    gpu_memory_usage_mb INTEGER[],
    gpu_utilization_percent DECIMAL(5, 2)[],
    disk_usage_percent DECIMAL(5, 2),
    network_io_mb DECIMAL(12, 3),
    temperature_celsius DECIMAL(5, 2),
    power_consumption_watts DECIMAL(8, 2)
);

-- Table for experiment tracking
CREATE TABLE IF NOT EXISTS multigrid.experiments (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    experiment_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    parameters JSONB NOT NULL,
    results JSONB,
    status VARCHAR(50) DEFAULT 'running',
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    created_by VARCHAR(100),
    tags TEXT[]
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_benchmark_results_name_time ON multigrid.benchmark_results(benchmark_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_size_solver ON multigrid.benchmark_results(problem_size, solver_type);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_platform_precision ON multigrid.benchmark_results(platform, precision_mode);
CREATE INDEX IF NOT EXISTS idx_validation_results_category_time ON multigrid.validation_results(test_category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_metrics_hostname_time ON multigrid.system_metrics(hostname, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_status_time ON multigrid.experiments(status, created_at DESC);

-- Create views for common queries
CREATE OR REPLACE VIEW multigrid.latest_benchmark_summary AS
SELECT 
    benchmark_name,
    problem_size,
    solver_type,
    precision_mode,
    platform,
    AVG(execution_time_seconds) as avg_execution_time,
    MIN(execution_time_seconds) as min_execution_time,
    MAX(execution_time_seconds) as max_execution_time,
    AVG(iterations_count) as avg_iterations,
    AVG(final_residual) as avg_final_residual,
    COUNT(*) as run_count,
    MAX(created_at) as last_run
FROM multigrid.benchmark_results 
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY benchmark_name, problem_size, solver_type, precision_mode, platform
ORDER BY last_run DESC;

CREATE OR REPLACE VIEW multigrid.performance_trends AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    solver_type,
    precision_mode,
    AVG(execution_time_seconds) as avg_execution_time,
    AVG(memory_usage_mb) as avg_memory_usage,
    AVG(energy_consumption_joules) as avg_energy_consumption,
    COUNT(*) as benchmark_count
FROM multigrid.benchmark_results
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY DATE_TRUNC('day', created_at), solver_type, precision_mode
ORDER BY date DESC, solver_type, precision_mode;

-- Create function to clean up old data
CREATE OR REPLACE FUNCTION multigrid.cleanup_old_data(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Delete old benchmark results
    DELETE FROM multigrid.benchmark_results 
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete old system metrics (keep less time due to volume)
    DELETE FROM multigrid.system_metrics 
    WHERE recorded_at < NOW() - INTERVAL '1 day' * LEAST(days_to_keep, 30);
    
    -- Clean up old validation results (keep longer)
    DELETE FROM multigrid.validation_results 
    WHERE created_at < NOW() - INTERVAL '1 day' * (days_to_keep * 2);
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions to multigrid user
GRANT ALL PRIVILEGES ON SCHEMA multigrid TO multigrid;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA multigrid TO multigrid;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA multigrid TO multigrid;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA multigrid TO multigrid;

-- Insert sample data for testing
INSERT INTO multigrid.benchmark_results (
    benchmark_name, problem_size, grid_dimensions, solver_type, 
    precision_mode, platform, execution_time_seconds, memory_usage_mb,
    iterations_count, final_residual, convergence_rate
) VALUES 
    ('poisson_2d', '513x513', '513x513', 'MultigridSolver', 'mixed', 'GPU', 1.234, 245, 8, 1.2e-10, 0.089),
    ('heat_equation', '257x257', '257x257', 'MultigridSolver', 'double', 'CPU', 2.456, 128, 12, 5.4e-9, 0.095),
    ('anisotropic', '1025x1025', '1025x1025', 'MultigridSolver', 'single', 'GPU', 4.567, 512, 15, 2.1e-8, 0.142);

-- Create a sample experiment
INSERT INTO multigrid.experiments (
    experiment_name, description, parameters, status
) VALUES (
    'convergence_study_2024', 
    'Comprehensive convergence analysis for different problem sizes',
    '{"grid_sizes": [64, 128, 256, 512], "solvers": ["Multigrid", "Jacobi"], "precisions": ["single", "double", "mixed"]}',
    'completed'
);

COMMIT;