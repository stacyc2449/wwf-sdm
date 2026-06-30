library(terra)
library(dismo)

ncr_raw <- read.csv("northern_corn_rootworm_0003360-260623161305970\\ncr.csv")
ncr_raw <- ncr_raw[which(ncr_raw$countryCode=="US"),] 