Sys.setenv(OMP_NUM_THREADS = "4")
Sys.setenv(OPENBLAS_NUM_THREADS = "4")
Sys.setenv(MKL_NUM_THREADS = "4")
Sys.setenv(VECLIB_MAXIMUM_THREADS = "4")

library(terra)
library(geodata)
library(dismo)
library(sf)
library(ggplot2)
library(lwgeom)
library(usmap)
library(rnaturalearth)

# arguments: raw distribution file, prevalence (0, 1, 2), name
args <- commandArgs(trailingOnly = TRUE)

name <- args[2]

# preprocessing: US, low coordinate uncertainty, within worldclim time frame
ncr_raw <- read.delim(args[1], sep="\t")
ncr_raw <- ncr_raw[which(ncr_raw$countryCode %in% c("US", "CA")), ]
ncr_raw <- ncr_raw[which(ncr_raw$year > 1970), ]
ncr_raw <- ncr_raw[which(!is.na(ncr_raw$decimalLatitude)), ]
ncr_raw <- ncr_raw[which(!is.na(ncr_raw$decimalLongitude)), ]
ncr_raw <- ncr_raw[which(ncr_raw$coordinateUncertaintyInMeters < 200000), ]

ncr <- data.frame(matrix(ncol = 2, nrow = length(ncr_raw$decimalLongitude)))
ncr[,1] <- ncr_raw$decimalLongitude
ncr[,2] <- ncr_raw$decimalLatitude

# study area extent for ncr presences
# e <- ext(-125, -66.9, 24.5, 49.4)
e <- ext(-170.0, -52.0, 25.0, 83.5)

ncr <- unique(ncr) # remove duplicates
ncr <- ncr[complete.cases(ncr),] # remove na's
colnames(ncr) <- c('lon','lat')

ncr <- ncr[which(ncr$lon>=e[1] & ncr$lon<=e[2]),]
ncr <- ncr[which(ncr$lat>=e[3] & ncr$lat<=e[4]),]

ncr_sf <- st_as_sf(ncr, coords = c("lon", "lat"), crs = 4326)

#crs 102009 corresponds to us projection, in meters
ncr_proj <- st_transform(ncr_sf, crs = 102009)

ncr_excl <- st_buffer(ncr_proj, dist = 100000) # distance of 200 km

ncr_excl_zone <- st_union(ncr_excl)
ncr_excl_zone <- st_make_valid(ncr_excl_zone)

# us_count <- us_map(regions = "states", exclude = c("Alaska", "Hawaii"))
states_provinces <- ne_states(
  country = c("United States of America", "Canada"),
  returnclass = "sf"
)

cont_us <- st_make_valid(states_provinces)
us_proj <- st_transform(cont_us, crs = 102009)

allowed_area <- st_difference(us_proj, ncr_excl_zone)

set.seed(42)
bg_high <- st_sample(allowed_area, size = 10000)
bg_mid <- st_sample(allowed_area, size = 1000)


print("points generated.")

bg_mid <- st_transform(bg_mid, crs = 4326)

bg_high <- st_transform(bg_high, crs = 4326)

bg_mid <- as.data.frame(st_coordinates(bg_mid))
bg_high <- as.data.frame(st_coordinates(bg_high))

allowed_area <- st_transform(allowed_area, crs = 4326)

# ncr is only lat/lon readings

file_list <- list.files(path = "worldclim", pattern = "\\.tif$", full.names = TRUE)
multilayer_raster <- rast(file_list)

# bioclim.data <- rast("chelsa_clim\\CHELSA_bio02_1981-2010_V.2.1.tif")
ras_ras <- crop(multilayer_raster, e*1.25)  # crop to bg point extent

colnames(bg_mid) <- c('lon','lat')
colnames(bg_high) <- c('lon', 'lat')

train_mid <- rbind(ncr, bg_mid)  # combine with presences
train_high <- rbind(ncr, bg_high)

# print(nrow(ncr))
pa_train_mid <- c(rep(1, nrow(ncr)), rep(0, nrow(bg_mid))) # col of ones and zeros
pa_train_high <- c(rep(1, nrow(ncr)), rep(0, nrow(bg_high)))
# print(bg_mid)

#final dataframe
train_mid <- data.frame(cbind(CLASS=pa_train_mid, train_mid))
train_high <- data.frame(cbind(CLASS=pa_train_high, train_high))

# create spatial points
crs <- crs(raster(ras_ras))
train_mid <- train_mid[sample(nrow(train_mid)),]
train_high <- train_high[sample(nrow(train_high)), ]


class.pa_mid <- data.frame(train_mid[,1])
colnames(class.pa_mid) <- 'CLASS'
dataMap.ncr_mid  <- SpatialPointsDataFrame(train_mid[,c(2,3)], class.pa_mid,
                                      proj4string =crs)

                                      # write as shp
dir.create(paste0("data/", name))
sf_map.ncr_mid <- st_as_sf(dataMap.ncr_mid, c("lon", "lat"), crs = 4326)
st_write(sf_map.ncr_mid, paste0('data/', name, '/mid.shp'), paste0(name, '_mid'), driver='ESRI Shapefile', append = FALSE)

class.pa_high <- data.frame(train_high[,1])
colnames(class.pa_high) <- 'CLASS'
dataMap.ncr_high  <- SpatialPointsDataFrame(train_high[,c(2,3)], class.pa_high,
                                      proj4string =crs)

                                      # write as shp
sf_map.ncr_high <- st_as_sf(dataMap.ncr_high, c("lon", "lat"), crs = 4326)
st_write(sf_map.ncr_high, paste0('data/', name, '/high.shp'), paste0(name, '_high'), driver='ESRI Shapefile', append = FALSE)



# ALL THE LOW THINGS, SINCE THIS MUST BE LOOPED 10 TIMES
dir.create(paste0("data/", name, "/low"))
for (i in 1:10){
    bg_low <- st_sample(allowed_area, size = 100)
    bg_low <- st_transform(bg_low, crs = 4326)
    bg_low <- as.data.frame(st_coordinates(bg_low))
    colnames(bg_low) <- c('lon', 'lat')
    train_low <- rbind(ncr, bg_low)
    pa_train_low <- c(rep(1, nrow(ncr)), rep(0, nrow(bg_low)))
    train_low <- data.frame(cbind(CLASS=pa_train_low, train_low))

    train_low <- train_low[sample(nrow(train_low)), ]

    class.pa_low <- data.frame(train_low[,1])
    colnames(class.pa_low) <- 'CLASS'
    dataMap.ncr_low  <- SpatialPointsDataFrame(train_low[,c(2,3)], class.pa_low,
                                        proj4string =crs)

                                        # write as shp
    sf_map.ncr_low <- st_as_sf(dataMap.ncr_low, c("lon", "lat"), crs = 4326)
    st_write(sf_map.ncr_low, paste0('data/', name, "/low/", i, '.shp'), paste0(name, '_low'), driver='ESRI Shapefile', append = FALSE)
}



# # plot our points
plot(ras_ras, main='NCR Presence and Absence')
points(bg_mid, col='red', pch = 16, cex = .3)
points(ncr, col='black', pch = 16, cex = .3)