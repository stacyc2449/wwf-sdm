library(ncdf4)
library(CFtime)
library(lattice)
library(RColorBrewer)

path <- "nc_files/added-tree-cover_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-e_gn_2100-2300.nc"
dname <- "added_tree_cover"

ncin <- nc_open(path)
print(ncin)

lon <- ncvar_get(ncin, "lon")
nlon <- dim(lon)

lat <- ncvar_get(ncin, "lat")
nlat <- dim(lat)

print(c(nlon, nlat))

time <- ncvar_get(ncin, "time")
tunits <- ncatt_get(ncin, "time", "units")
print(tunits)

base_date <- as.Date("2100-01-01")

nt <- dim(time)
print(nt)


tmp_array <- ncvar_get(ncin, dname)
dlname <- ncatt_get(ncin, dname, "long_name")
dunits <- ncatt_get(ncin, dname, "units")
fillvalue <- ncatt_get(ncin, dname, "_FillValue")
dim(tmp_array)

title <- ncatt_get(ncin,0,"title")
institution <- ncatt_get(ncin,0,"institution")
datasource <- ncatt_get(ncin,0,"source")
references <- ncatt_get(ncin,0,"references")
history <- ncatt_get(ncin,0,"history")
conventions <- ncatt_get(ncin,0,"Conventions")

cf <- CFtime(tunits$value, calendar = "proleptic_gregorian", time) # convert time to CFtime class
# print(cf)

timestamps <- as_timestamp(cf) # get character-string times
# print(timestamps)

dates <- as.Date(substr(timestamps, 1, 10))
# timecf <- parse_timestamps(cf, dates)

tmp_array[tmp_array==fillvalue$value] <- NA
print(length(na.omit(as.vector(tmp_array[,,1]))))

m <- 10
tmp_slice <- tmp_array[,,m]

cat("Dimensions of tmp_slice:\n")
print(dim(tmp_slice))

cat("Range before cleaning:\n")
print(range(tmp_slice, na.rm = TRUE))

cat("Number of NA values:\n")
print(sum(is.na(tmp_slice)))

cat("Number of finite values:\n")
print(sum(is.finite(tmp_slice)))

lon_order <- order(lon)
lat_order <- order(lat)

lon_sorted <- lon[lon_order]
lat_sorted <- lat[lat_order]

tmp_sorted <- tmp_slice[lon_order, lat_order]

image(lon_sorted, lat_sorted, tmp_sorted, col=rev(brewer.pal(10,"RdBu")))