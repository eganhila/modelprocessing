import numpy as np
import netCDF4 as nc

nc_data = nc.Dataset("/Volumes/triton/Data/ModelChallenge/MGITM/MGITM_LS270_F200_150615.nc")

for key, var in nc_data.variables.items():
    print key, var.shape

nc_data.close()
