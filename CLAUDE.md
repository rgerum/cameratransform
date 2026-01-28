# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CameraTransform is a Python library for fitting camera transformations and projecting points between camera space (2D image coordinates) and world space (3D coordinates). It supports multiple projection models, lens distortion correction, and camera parameter fitting via Bayesian methods.

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync --locked --all-extras --dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_transforms.py

# Run a specific test
uv run pytest tests/test_transforms.py::TestTransforms::test_init_cam

# Build documentation (requires docs dependencies)
uv sync --group docs
cd docs && make html
```

## Architecture

### Core Components

The library is built around a composable camera model with three main components:

1. **Projection** (`projection.py`) - Maps 3D camera coordinates to 2D image coordinates
   - `RectilinearProjection` - Standard pinhole/frame camera model
   - `CylindricalProjection` - For panoramic images (360° horizontal)
   - `EquirectangularProjection` - Full spherical panoramic projection

2. **SpatialOrientation** (`spatial.py`) - Defines camera position and rotation in world space
   - Uses ZXZ Euler angles: heading, tilt, roll
   - Position: pos_x_m, pos_y_m, elevation_m
   - Alternative `SpatialOrientationYawPitchRoll` class available

3. **LensDistortion** (`lens_distortion.py`) - Radial lens distortion correction
   - `NoDistortion` - Default, no correction
   - `BrownLensDistortion` - k1, k2, k3 coefficients (OpenCV compatible)
   - `ABCDistortion` - a, b, c, d polynomial model (PTGui compatible)

### Camera Class (`camera.py`)

The `Camera` class combines projection, orientation, and lens distortion:

```python
cam = ct.Camera(
    ct.RectilinearProjection(focallength_px=3729, image=(4608, 2592)),
    ct.SpatialOrientation(elevation_m=15.4, tilt_deg=85),
    ct.BrownLensDistortion(k1=0.01)
)
```

Key methods:
- `imageFromSpace(points)` - Project 3D world points to 2D image
- `spaceFromImage(points, Z=0)` - Back-project 2D image points to 3D (requires constraint)
- `getRay(points)` - Get ray direction for image points
- `getTopViewOfImage(image)` - Generate bird's-eye view projection

### Parameter System (`parameter_set.py`)

All components use `ParameterSet` for parameter management:
- Parameters can be user-set or fitted
- Supports Bayesian fitting via `metropolis()` and `fridge()` methods
- `addObjectHeightInformation()`, `addLandmarkInformation()`, `addHorizonInformation()` add probability terms for fitting

### CameraGroup (`camera.py`)

For stereo vision with multiple cameras sharing parameters:
- `spaceFromImages(points1, points2)` - Triangulate 3D points from two views
- `imagesFromSpace(points)` - Project to all cameras

### Coordinate Systems

- **Image coordinates**: 2D pixel positions (x, y)
- **Camera coordinates**: 3D, camera at origin, -Z pointing forward
- **Space coordinates**: 3D world coordinates (X, Y, Z)
- **GPS coordinates**: Latitude, longitude, elevation

## Testing

Tests use `hypothesis` for property-based testing with custom strategies in `tests/strategies.py`. The test suite mocks optional dependencies (PIL, cv2, etc.) to allow running without them installed.
