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

if(args[2] == "0"){
    bg_sample <- 0.1
    bg_rep <- TRUE
} else if(args[2] == "2"){
    bg_sample <- 10
    bg_rep <- FALSE
} else {
    bg_sample <- 1
    bg_rep <- FALSE
}

name <- args[3]

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
bg <- st_sample(allowed_area, size = 1000)
print("points generated.")
bg <- st_transform(bg, crs = 4326)
bg <- as.data.frame(st_coordinates(bg))

allowed_area <- st_transform(allowed_area, crs = 4326)

# ncr is only lat/lon readings

file_list <- list.files(path = "worldclim", pattern = "\\.tif$", full.names = TRUE)
multilayer_raster <- rast(file_list)

bioclim.data <- rast("chelsa_clim\\CHELSA_bio02_1981-2010_V.2.1.tif")
ras_ras <- crop(multilayer_raster, e*1.25)  # crop to bg point extent

colnames(bg) <- c('lon','lat')
train <- rbind(ncr, bg)  # combine with presences
print(nrow(ncr))
pa_train <- c(rep(1, nrow(ncr)), rep(0, nrow(bg))) # col of ones and zeros
print(bg)

#final dataframe
train <- data.frame(cbind(CLASS=pa_train, train))

# create spatial points
crs <- crs(raster(ras_ras))
train <- train[sample(nrow(train)),]
class.pa <- data.frame(train[,1])
colnames(class.pa) <- 'CLASS'
dataMap.ncr  <- SpatialPointsDataFrame(train[,c(2,3)], class.pa,
                                      proj4string =crs)

                                      # write as shp
sf_map.ncr <- st_as_sf(dataMap.ncr, c("lon", "lat"), crs = 4326)
st_write(sf_map.ncr, 'data/ncr.shp', 'ncr', driver='ESRI Shapefile', append = FALSE)

# # plot our points
plot(bioclim.data, main='NCR Presence and Absence')
points(bg, col='red', pch = 16, cex = .3)
points(ncr, col='black', pch = 16, cex = .3)