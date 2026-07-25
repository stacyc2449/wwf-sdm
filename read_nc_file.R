library(ncdf4)

path <- "nc_files/added-tree-cover_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-e_gn_2100-2300.nc"
ncin <- nc_open(path)
print(ncin)

lon <- ncvar_get(ncin,"lon_bounds")
nlon <- dim(lon)
head(lon)