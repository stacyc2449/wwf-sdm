# # SDM in Python

# ### Generate Pseudo-Absence Data
# Absence data must be generated for these classifiers, and different quantities must be generated for types of classifiers, based on Barbet-Massin et al., 2012.

import os
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pygam import LogisticGAM, s, f
from sklearn.tree import DecisionTreeClassifier
# out of all the glms, we choose the poisson - it does not have negative values, and varying variance???
# also we are looking for a scale of values up to 1000...
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pyimpute import impute
from sklearn import model_selection
from elapid import MaxentModel

import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import shutil
import glob
from pyimpute import load_training_vector
from pyimpute import load_targets
import numpy as np
import pandas as pd
import rasterio

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.frozen import FrozenEstimator
import heapq

from pylab import plt
from rasterio.windows import from_bounds
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.ensemble import VotingClassifier
from rasterio.enums import Resampling
import xarray as xr
import regionmask
import logging

# logging
logging.basicConfig(
    filename='corr.log',
    filemode='a',               # 'a' to append logs, 'w' to overwrite every run
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO          # Capture INFO, WARNING, ERROR, and CRITICAL logs
)

files_list = ["inputs\\nst_pest_sightings\\bean_leaf_beetle_0003221-260623161305970\\0003221-260623161305970.csv", 
              "inputs\\nst_pest_sightings\\bird_cherry_aphid_0003207-260623161305970\\0003207-260623161305970.csv",
              "inputs\\nst_pest_sightings\\black_cutworm_0003212-260623161305970\\0003212-260623161305970.csv",
            #   "nst_pest_sightings\\corn_leaf_aphid_0003215-260623161305970\\0003215-260623161305970.csv",
              "inputs\\nst_pest_sightings\\differential_grasshopper0032387-260623161305970\\0032387-260623161305970.csv",
            #   "nst_pest_sightings\\european_corn_borer0003385-260623161305970\\0003385-260623161305970.csv",
              "inputs\\nst_pest_sightings\\green_stink_0005409-260623161305970\\0005409-260623161305970.csv",
            #   "nst_pest_sightings\\hessian_fly_0003394-260623161305970\\0003394-260623161305970.csv",
              "inputs\\nst_pest_sightings\\japanese_beetle_0003383-260623161305970\\0003383-260623161305970.csv",
              "inputs\\nst_pest_sightings\\northern_corn_rootworm_0003360-260623161305970\\0003360-260623161305970.csv",
            #   "nst_pest_sightings\\reg_legged_grasshopperobservations-752840.csv\\observations-752840.csv",
              "inputs\\nst_pest_sightings\\seedcorn_maggot_0003200-260623161305970\\0003200-260623161305970.csv",
              "inputs\\nst_pest_sightings\\southern_green_stink_0003348-260623161305970\\0003348-260623161305970.csv",
              "inputs\\nst_pest_sightings\\three_cornered_alfalfa_0003390-260623161305970\\0003390-260623161305970.csv",
              "inputs\\nst_pest_sightings\\true_armyworm0032387-260623161305970\\0032387-260623161305970.csv",
              "input\\nst_pest_sightings\\two_striped_grasshopper0032397-260623161305970\\0032397-260623161305970.csv",
              "inputs\\nst_pest_sightings\\western_corn_rootworm_0003372-260623161305970\\0003372-260623161305970.csv"
              ]

# Run the subprocess, R code to generate pseudo absences. Will be customized depending on the models
def make_model_dict(name, model, model_class):
    return {
        "name": name,
        "model": model,
        "threshold": -1,
        "auc": -1,
        "maxsss": -1,
        "frozen model": None,
        "class": model_class
    }

####
#### MAKE SURE TO CHANGE FOR CUSTOM R FILE PATH
####
def make_cmd(path, name):
    return ["C:\\Program Files\\R\\R-4.4.1\\bin\\Rscript.exe",'pseudo_absence.R', 
           path, name]

def plotit(x, title, colors, cmap="Blues"):
    x = np.asarray(x)
    unique_vals = [0, 1, -9999]
    n_val = len(unique_vals)
    base = plt.get_cmap(cmap, n_val)
    plt.imshow(x, cmap=cmap, interpolation='nearest')
    
    x_mapped = np.full(x.shape, np.nan)

    for i, val in enumerate(unique_vals):
        x_mapped[x == val] = i
    
    discrete_cmap = ListedColormap(colors)

    bounds = np.arange(-0.5, n_val + 0.5, 1)
    norm = BoundaryNorm(bounds, discrete_cmap.N)

    im = plt.imshow(
        x_mapped,
        cmap=discrete_cmap,
        norm=norm,
        interpolation="nearest"
    )

    cbar = plt.colorbar(im, ticks=np.arange(n_val))
    cbar.ax.set_yticklabels(unique_vals)

    plt.title(title, fontweight="bold")
    plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.show()


command_blb = make_cmd(files_list[0], "beanleafbeetle")
command_bca = make_cmd(files_list[1], "bircherryaphid")
command_bc = make_cmd(files_list[2], "blackcutworm")
command_dg = make_cmd(files_list[3], "differentialgrasshopper") 
command_gs = make_cmd(files_list[4], "greenStink")
command_jb = make_cmd(files_list[5], "japanesebeetle")
command_ncr = make_cmd(files_list[6], "northerncornrootworm")
command_sm = make_cmd(files_list[7], "seedcornmaggot")
command_sgs = make_cmd(files_list[8], "southerngreenstink")
command_tca = make_cmd(files_list[9], "threecornerneredalfalfa")
command_ta = make_cmd(files_list[10], "truearmyworm")
command_tsg = make_cmd(files_list[11], "two-striped-grasshopper")
command_wcr = make_cmd(files_list[12], "westcornrootworm")

command_list = [command_blb, command_bca, command_bc, command_dg, command_gs, command_jb, command_ncr, command_sm, command_sgs, command_tca, command_ta, command_tsg, command_wcr]

# THE ORDER IS SIGNIFICANT!!!!
ras_feats = ["inputs/chelsa_clim/current/CHELSA_bio02_1981-2010_V.2.1.tif", 
             "inputs/chelsa_clim/current/CHELSA_bio04_1981-2010_V.2.1.tif", 
             "inputs/chelsa_clim/current/CHELSA_bio06_1981-2010_V.2.1.tif",
             "inputs/chelsa_clim/current/CHELSA_bio14_1981-2010_V.2.1.tif",
             "inputs/chelsa_clim/current/CHELSA_bio15_1981-2010_V.2.1.tif",
             "inputs/chelsa_clim/current/CHELSA_bio19_1981-2010_V.2.1.tif",
             "inputs/chelsa_clim/current/CHELSA_fgd_1981-2010_V.2.1.tif",
             "inputs/chelsa_clim/current/CHELSA_scd_1981-2010_V.2.1.tif"]

print('There are ', len(ras_feats), ' raster features.')
logging.info("Current raster features loaded: ".join(ras_feats))

# loading future prediction


for command in command_list:
    logging.info(command[3])

    # ### Converting shapefiles (.shp) to geopandas dataframe

    ncr_gdf_mid = gpd.GeoDataFrame.from_file('outputs/data/' + command[3] + '/mid.shp')
    # ncr_gdf_high = gpd.GeoDataFrame.from_file('data/' + command[3] + '/high.shp')

    # Checking duplicates and NA values. Coordinate reference system should ideally be epsg: 4326
    print("Coordinate reference system: {}".format(ncr_gdf_mid.crs))
    print("{} observations with {} columns".format(*ncr_gdf_mid.shape))

    logging.info("{} observations with {} columns".format(*ncr_gdf_mid.shape))

    with rasterio.open(ras_feats[0]) as src:
        raster_crs = src.crs

    ncr_gdf_mid = ncr_gdf_mid.to_crs(raster_crs)

    
    print(ncr_gdf_mid.crs)

    # Generate raster maps of suitability. Also separate test sets from the rest for cross validation

    train_xs_mid, train_ys_mid = load_training_vector(ncr_gdf_mid, ras_feats, response_field='CLASS')

    x_df_mid = pd.DataFrame(train_xs_mid)
    x_df_mid = x_df_mid.apply(pd.to_numeric, errors="coerce")
    x_df_mid = x_df_mid.replace([np.inf, -np.inf], np.nan)

    y_ser_mid = pd.Series(train_ys_mid)

    mask_mid = x_df_mid.notna().all(axis=1) & y_ser_mid.notna()

    # Apply mask_mid to both X and y
    train_xs_mid_clean = x_df_mid.loc[mask_mid].to_numpy(dtype=float)
    train_ys_mid_clean = y_ser_mid.loc[mask_mid].to_numpy()

    corr = x_df_mid.corr(method="pearson")
    logging.info(corr.to_string())