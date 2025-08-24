"""
Version information for mixed-precision multigrid package.
"""

# Version follows semantic versioning: MAJOR.MINOR.PATCH
# MAJOR: Breaking changes to API
# MINOR: New features, backwards compatible
# PATCH: Bug fixes, backwards compatible

__version__ = "1.0.0"

# Version components for programmatic access
VERSION_INFO = tuple(int(x) for x in __version__.split('.'))

# Development status
DEV_STATUS = "stable"  # alpha, beta, rc, stable

# Build metadata (will be updated by CI/CD)
BUILD_DATE = None
BUILD_HASH = None
BUILD_BRANCH = None

def get_version_info():
    """
    Get comprehensive version information.
    
    Returns:
        dict: Version information including version string, components, 
              development status, and build metadata
    """
    return {
        "version": __version__,
        "version_info": VERSION_INFO,
        "dev_status": DEV_STATUS,
        "build_date": BUILD_DATE,
        "build_hash": BUILD_HASH,
        "build_branch": BUILD_BRANCH,
    }

def is_stable_release():
    """Check if this is a stable release."""
    return DEV_STATUS == "stable" and "dev" not in __version__

def requires_python_version():
    """Get minimum required Python version."""
    return (3, 8)  # Python 3.8+

def check_python_version():
    """Check if current Python version meets requirements."""
    import sys
    required = requires_python_version()
    current = sys.version_info[:2]
    return current >= required