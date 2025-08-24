#!/bin/bash

echo "🔍 Mixed-Precision Multigrid Project Verification"
echo "================================================="

# Test 1: Basic Import Test
echo "1. Testing core imports..."
python3 -c "
try:
    from src.multigrid.core.grid import Grid
    from src.multigrid.solvers.multigrid import MultigridSolver
    from src.multigrid.applications.poisson_solver import PoissonSolver2D
    print('✅ Core imports: SUCCESS')
except Exception as e:
    print(f'❌ Import error: {e}')
"

# Test 2: Basic Functionality Test
echo "2. Testing basic solver functionality..."
python3 -c "
try:
    import numpy as np
    from src.multigrid.applications.poisson_solver import PoissonSolver2D
    
    # Simple solver test
    solver = PoissonSolver2D()
    print('✅ Solver creation: SUCCESS')
except Exception as e:
    print(f'❌ Solver error: {e}')
"

# Test 3: Run Quick Tests
echo "3. Running pytest on core modules..."
python3 -m pytest tests/unit/ -v --tb=short -x

# Test 4: Check GPU Availability
echo "4. Checking GPU support..."
python3 -c "
try:
    import cupy as cp
    print('✅ CuPy available')
    print(f'   GPU devices: {cp.cuda.runtime.getDeviceCount()}')
except:
    print('⚠️  CuPy not available (CPU-only mode)')
"

# Test 5: Test Tutorial Notebooks
echo "5. Checking tutorial notebooks..."
if [ -d "examples" ]; then
    echo "✅ Tutorial notebooks present:"
    ls examples/*.ipynb | head -3
else
    echo "❌ No tutorial notebooks found"
fi

# Test 6: Jekyll Website Test
echo "6. Testing documentation build..."
if [ -f "docs/_config.yml" ]; then
    echo "✅ Jekyll website configured"
    cd docs && bundle install --quiet && bundle exec jekyll build --quiet
    if [ $? -eq 0 ]; then
        echo "✅ Website builds successfully"
    else
        echo "⚠️  Website build issues"
    fi
    cd ..
else
    echo "❌ Jekyll not configured"
fi

echo "================================================="
echo "Verification complete!"