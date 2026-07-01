library(terra)
library(geodata)
library(dismo)
library(sf)
library(tigris)
library(ggplot2)

ncr_raw <- read.delim("northern_corn_rootworm_0003360-260623161305970\\ncr.csv", sep="\t")
ncr_raw <- ncr_raw[which(ncr_raw$countryCode=="US"),]
ncr <- data.frame(matrix(ncol = 2, nrow = length(ncr_raw$decimalLongitude)))
ncr[,1] <- ncr_raw$decimalLongitude
ncr[,2] <- ncr_raw$decimalLatitude

e <- ext(-125, -66.9, 24.5, 49.4) # set study area extent

ncr <- unique(ncr) # remove duplicates
ncr <- ncr[complete.cases(ncr),] # remove na's
colnames(ncr) <- c('lon','lat')

ncr <- ncr[which(ncr$lon>=e[1] & ncr$lon<=e[2]),]
ncr <- ncr[which(ncr$lat>=e[3] & ncr$lat<=e[4]),]


us_count <- ne_countries(country = "united states of america", scale = "medium", returnclass = "sf")
# continental <- us_count %>%
#   filter(!name %in% c("Alaska", "Hawaii"))

set.seed(42)
bg <- st_sample(us_count, size = 100)
bg <- as.data.frame(st_coordinates(bg))

# continental_us <- subset(usa_all, !usa_all$NAME_1 %in% c("Alaska", "Hawaii"))

# ncr is only lat/lon readings

# data is all manually loaded, using CHELSA data, because geodata server is down :(

file_list <- list.files(path = "worldclim", pattern = "\\.tif$", full.names = TRUE)
multilayer_raster <- rast(file_list)

# bioclim.data <- rast("chelsa_clim\\CHELSA_bio02_1981-2010_V.2.1.tif")
ras_ras <- crop(bioclim.data, e*1.25)  # crop to bg point extent


# bg <- randomPoints(raster(ras_ras), 100, ncr, ext=gen_e, extf = 1.25) 
print("points generated.")
colnames(bg) <- c('lon','lat')
train <- rbind(ncr, bg)  # combine with presences
print(nrow(ncr))
pa_train <- c(rep(1, nrow(ncr)), rep(0, nrow(bg))) # col of ones and zeros
print(nrow(bg))

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
sf_map.ncr = st_as_sf(dataMap.ncr, c("lon", "lat"), crs = 4326)
st_write(sf_map.ncr, 'data/ncr.shp', 'ncr', driver='ESRI Shapefile', append = FALSE)

# # plot our points
plot(ras_ras, main='NCR Presence and Absence')
points(bg, col='red', pch = 16,cex=.3)
points(ncr, col='black', pch = 16,cex=.3)
plot(wrld_simpl, add=TRUE, border='dark grey')