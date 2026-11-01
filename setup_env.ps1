$ErrorActionPreference = "Stop"
$env:UV_NATIVE_TLS = "1"
Set-Location c:\dev\research\austin-twin

Write-Output "=== 1. install cpython 3.12 ==="
python -m uv python install 3.12

Write-Output "=== 2. create venv ==="
python -m uv venv --python 3.12 .venv

$py = "c:\dev\research\austin-twin\.venv\Scripts\python.exe"
Write-Output "=== venv python ==="
& $py --version

Write-Output "=== 3. install project deps ==="
python -m uv pip install --python $py -e .

Write-Output "=== 4. extra deps ==="
python -m uv pip install --python $py netcdf4 cdsapi pytest zarr "dask[array]" gcsfs s3fs

Write-Output "=== 5. install torch (cu128, Blackwell sm_120) ==="
python -m uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu128

Write-Output "=== 6. verify ==="
& $py -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available());print('dev',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE');print('cap',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')"
& $py -c "import numpy,scipy,xarray,rasterio,geopandas,osmnx,pystac_client;print('geostack OK')"
Write-Output "=== DONE ==="
