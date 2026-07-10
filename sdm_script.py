# Moving onto a python file for better readability, and because this requires large for-loops

import os
# os.mkdir("inputs", ex)
# os.mkdir("outputs")
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import ExtraTreesClassifier 
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

import geopandas as gpd
import shutil
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import rasterio



files_list = ["nst_pest_sightings\\bean_leaf_beetle_0003221-260623161305970\\0003221-260623161305970.csv", 
              "nst_pest_sightings\\bird_cherry_aphid_0003207-260623161305970\\0003207-260623161305970.csv",
              "nst_pest_sightings\\black_cutworm_0003212-260623161305970\\0003212-260623161305970.csv",
            #   "nst_pest_sightings\\corn_leaf_aphid_0003215-260623161305970\\0003215-260623161305970.csv",
              "nst_pest_sightings\\differential_grasshopper_observations-752834.csv\\observations-752834.csv",
            #   "nst_pest_sightings\\european_corn_borer0003385-260623161305970\\0003385-260623161305970.csv",
              "nst_pest_sightings\\green_stink_0005409-260623161305970\\0005409-260623161305970.csv",
            #   "nst_pest_sightings\\hessian_fly_0003394-260623161305970\\0003394-260623161305970.csv",
              "nst_pest_sightings\\japanese_beetle_0003383-260623161305970\\0003383-260623161305970.csv",
              "nst_pest_sightings\\northern_corn_rootworm_0003360-260623161305970\\0003360-260623161305970.csv",
            #   "nst_pest_sightings\\reg_legged_grasshopperobservations-752840.csv\\observations-752840.csv",
              "nst_pest_sightings\\seedcorn_maggot_0003200-260623161305970\\0003200-260623161305970.csv",
              "nst_pest_sightings\\southern_green_stink_0003348-260623161305970\\0003348-260623161305970.csv",
              "nst_pest_sightings\\three_cornered_alfalfa_0003390-260623161305970\\0003390-260623161305970.csv",
              "nst_pest_sightings\\true_armyworm\\armyworm_moth_observations-752830.csv",
            #   "nst_pest_sightings\\two_striped_grasshopper_observations-752841.csv\\observations-752841.csv",
              "nst_pest_sightings\\western_corn_rootworm_0003372-260623161305970\\0003372-260623161305970.csv"
              ]

for file in files_list:
    with open(file, mode='r', encoding='utf-8') as file:
        row_count = sum(1 for line in file)

    print(f"{file}, Total lines: {row_count}")

command_blb = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[0], "2", "northerncorn"]

command_bca = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[1], "2", "birdcherryaphid"]

command_bc = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "blackcutworm"]

command_dg = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "differentialgrasshopper"]

command_gs = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "greenStink"]

command_jb = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "japanesebeetle"]

command_ncr = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "northerncornrootworm"]

command_sm = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "seedcornmaggot"]

command_sgs = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "southerngreenstink"]

command_tca = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "threecornerneredalfalfa"]

command_ta = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "truearmyworm"]

command_wcr = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "2", "westcornrootworm"]

command_list = [command_blb, command_bca, command_bc, command_dg, command_gs, command_jb, command_ncr, command_sm, command_sgs, command_tca, command_ta, command_wcr]

for command in command_list:
    try:
        subprocess.run(command, capture_output=True, 
            text=True, check = True)
    except subprocess.CalledProcessError as e:
        print(f"R Script failed with exit code {e.returncode}")
        print("Error Message:\n", e.stderr)

ncr_gdf = gpd.GeoDataFrame.from_file('data/ncr.shp')
ncr_gdf.sample(5)

ncr_gdf[ncr_gdf.CLASS == 1].plot(marker='*', color='green', markersize=5)
ncr_gdf[ncr_gdf.CLASS == 0].plot(marker='*', color='red', markersize=5)

ras_feats = sorted(glob.glob(
    'worldclim/wc2.1_2.5m_bio_*.tif'))

# THE ORDER IS SIGNIFICANT!!!!

print('There are ', len(ras_feats), ' raster features.')
print(ras_feats)
# current order: 13, 14, 15, 2, 4, 5, 6

with rasterio.open(ras_feats[0]) as src:
    raster_crs = src.crs

ncr_gdf = ncr_gdf.to_crs(raster_crs)
 
print(ncr_gdf.crs)

from pyimpute import load_training_vector
from pyimpute import load_targets
import numpy as np
import pandas as pd

train_xs, train_ys = load_training_vector(ncr_gdf, ras_feats, response_field='CLASS')

x_df = pd.DataFrame(train_xs)
x_df = x_df.apply(pd.to_numeric, errors="coerce")
x_df = x_df.replace([np.inf, -np.inf], np.nan)

y_ser = pd.Series(train_ys)

mask = x_df.notna().all(axis=1) & y_ser.notna()

# Apply mask to both X and y
train_xs_clean = x_df.loc[mask].to_numpy(dtype=float)
train_ys_clean = y_ser.loc[mask].to_numpy()

print("Original samples:", len(train_xs))
print("Clean samples:", len(train_xs_clean))
print("Dropped samples:", len(train_xs) - len(train_xs_clean))


train_x, test_x, train_y, test_y = model_selection.train_test_split(train_xs_clean, train_ys_clean, test_size=0.25, random_state=42, stratify=train_ys_clean)
# train_xs.shape, train_ys.shape
train_x.shape, test_x.shape, train_y.shape, test_y.shape

print(train_x)

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score

sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)
k = 4
kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

### random forest ###

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1)
rf_accuracy_scores = model_selection.cross_val_score(rf, train_x, train_y, cv=kf, scoring='accuracy')
print("rf %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, rf_accuracy_scores.mean() * 100, rf_accuracy_scores.std() * 200))
rf.fit(train_x, train_y)
rf_train_pred = rf.predict(train_x)
rf_test_pred = rf.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, rf_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, rf_test_pred))

### LDA ###

lda = LinearDiscriminantAnalysis()
lda_accuracy_scores = model_selection.cross_val_score(lda, train_x, train_y, cv = kf, scoring='accuracy')
print("lda %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, lda_accuracy_scores.mean() * 100, lda_accuracy_scores.std() * 200))
lda.fit(train_x, train_y)
lda_train_pred = lda.predict(train_x)
lda_test_pred = lda.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, lda_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, lda_test_pred))

### decision tree classifier ###

dtc = DecisionTreeClassifier()
dtc_accuracy_scores = model_selection.cross_val_score(dtc, train_x, train_y, cv = kf, scoring='accuracy')
print("dtc %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, dtc_accuracy_scores.mean() * 100, dtc_accuracy_scores.std() * 200))
dtc.fit(train_x, train_y)
dtc_train_pred = dtc.predict(train_x)
dtc_test_pred = dtc.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, dtc_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, dtc_test_pred))

### logistic regression ###
lr = LogisticRegression(max_iter=5000)
lr_accuracy_scores = model_selection.cross_val_score(lr, train_x, train_y, cv = kf, scoring='accuracy')
print("lr %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, lr_accuracy_scores.mean() * 100, lr_accuracy_scores.std() * 200))
lr.fit(train_x, train_y)
lr_train_pred = lr.predict(train_x)
lr_test_pred = lr.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, lr_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, lr_test_pred))

### xgb classifier ###
xgb = XGBClassifier()
xgb_accuracy_scores = model_selection.cross_val_score(xgb, train_x, train_y, cv = kf, scoring='accuracy')
print("xgb %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, xgb_accuracy_scores.mean() * 100, xgb_accuracy_scores.std() * 200))
xgb.fit(train_x, train_y)
xgb_train_pred = xgb.predict(train_x)
xgb_test_pred = xgb.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, xgb_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, xgb_test_pred))

### light gbm ###
lgbm = LGBMClassifier()
lgbm_accuracy_scores = model_selection.cross_val_score(lgbm, train_x, train_y, cv = kf, scoring='accuracy')
print("lgbm %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, lgbm_accuracy_scores.mean() * 100, lgbm_accuracy_scores.std() * 200))
lgbm.fit(train_x, train_y)
lgbm_train_pred = lgbm.predict(train_x)
lgbm_test_pred = lgbm.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, lgbm_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, lgbm_test_pred))

## MARS ###
mars = make_pipeline(SplineTransformer(degree=1, n_knots=10, include_bias=False), LinearRegression())
mars_accuracy_scores = model_selection.cross_val_score(mars, train_x, train_y, cv = kf, scoring='accuracy')
print("mars %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
          % (k, mars_accuracy_scores.mean() * 100, mars_accuracy_scores.std() * 200))
mars.fit(train_x, train_y)
mars_train_pred = mars.predict(train_x)
mars_test_pred = mars.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, mars_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, mars_test_pred))




    # # 'et': (ExtraTreesClassifier()),
    # 'lda': (LinearDiscriminantAnalysis()),
    # 'dtc': (DecisionTreeClassifier()),
    # 'glm': (LogisticRegression()),
    # 'xgb': (XGBClassifier()),
    # 'lgbm': (LGBMClassifier()),
    # # 'maxent': (MaxentModel())

ALT_MAP = {
    # 'gam': (GAM()),
    'mars': (make_pipeline(SplineTransformer(degree=1, n_knots=10, include_bias=False), LinearRegression()))
}



# for name, (model) in CLASS_MAP.items():
#      # k-fold
#     # spatial prediction
#     score = model.score(test_x, test_y)
#     print(name + " score: %f" % (score))
    # impute(target_xs, model, raster_info, outdir='outputs/' + name + '-images',
    #        class_prob=True, certainty=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
acc_scores = []
logloss_scores = []

gam = LogisticGAM(s(0) + s(1) + s(2))

for train_idx, test_idx in cv.split(train_x, train_y):
    X_train, X_test = train_x[train_idx], train_x[test_idx]
    y_train, y_test = train_y[train_idx], train_y[test_idx]
    gam.fit(X_train, y_train)

    y_prob = gam.predict_proba(X_test)

    y_pred = gam.predict(X_test)

    auc_scores.append(roc_auc_score(y_test, y_prob))
    acc_scores.append(accuracy_score(y_test, y_pred))
    logloss_scores.append(log_loss(y_test, y_prob))

print("Mean AUC:", np.mean(auc_scores))
print("Mean accuracy:", np.mean(acc_scores))
print("Mean log loss:", np.mean(logloss_scores))


### gam ###
## pygam does not support cross val, find alternatives?


# gam_accuracy_scores = gam.gridsearch()
# model_selection.cross_val_score(gam, train_x, train_y, cv = kf, scoring='accuracy')
# print("gam %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
#           % (k, gam_accuracy_scores.mean() * 100, gam_accuracy_scores.std() * 200))

gam_train_pred = gam.predict(train_x)
gam_test_pred = gam.predict(test_x)
print("Train balanced accuracy:", balanced_accuracy_score(train_y, gam_train_pred))
print("Test balanced accuracy:", balanced_accuracy_score(test_y, gam_test_pred))

from pylab import plt
future_stacks = ["worldclim\\future\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp126_2021-2040.tif", "worldclim\\future\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp245_2021-2040.tif",
                "worldclim\\future\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp370_2021-2040.tif", "worldclim\\future\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp585_2021-2040.tif"]
future_40_stacks = ["worldclim\\future40\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp126_2041-2060.tif", "worldclim\\future40\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp245_2041-2060.tif", "worldclim\\future40\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp370_2041-2060.tif",
                    "worldclim\\future40\\wc2.1_2.5m_bioc_GISS-E2-1-G_ssp585_2041-2060.tif"]
out_dir = "future_split_bands"

from rasterio.windows import from_bounds

def plotit(x, title, cmap="Blues"):
    plt.imshow(x, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title(title, fontweight = 'bold')


training_feature_names = [
    "bio13",
    "bio14",
    "bio15",
    "bio2",
    "bio4",
    "bio5",
    "bio6"
]

future_band_map = {
    "bio13": 13,
    "bio14": 14,
    "bio15": 15,
    "bio2": 2,
    "bio4": 4,
    "bio5": 5,
    "bio6": 6
}
min_lon = -170
max_lon = -52
min_lat = 24
max_lat = 83.5

future_ras_feats = []

for index, stack in enumerate(future_40_stacks):
    # os.mkdir(f"{out_dir}_{index}")
    with rasterio.open(stack) as src:
        # print("Band count:", src.count)
        # print("CRS:", src.crs)
        # print("Shape:", src.shape)
        # print("Bounds:", src.bounds)
        # print("Descriptions:", src.descriptions)
        # print("Indexes:", src.indexes)
# with rasterio.open(future_stack) as src:
        window = from_bounds(
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                transform=src.transform
        )
        data = src.read(window=window)

        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(height=data.shape[1],
                width=data.shape[2],
                transform=transform)
        
        temp_future_ras_feats = []
        for feature_name in training_feature_names:
            
            band_index = future_band_map[feature_name]

            band = src.read(band_index)

            out_path = os.path.join(f"{out_dir}_{index}", f"future_{feature_name}.tif")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(band, 1)
                dst.set_band_description(1, feature_name)

            temp_future_ras_feats.append(out_path)
            print(temp_future_ras_feats)

    future_ras_feats.append(temp_future_ras_feats)
    # future_ras_feats = sorted(glob.glob(
    #     'future_split_bands_1/future_bio*.tif')) 

print(future_ras_feats)

ssp = ['126', '245', '370', '585']
future_target_xss = []
future_raster_infos = []
for index, feat in enumerate(future_ras_feats):
    future_target_xs, future_raster_info = load_targets(feat)

    impute(future_target_xs, rf, future_raster_info, outdir='outputs/' + 'rf1' + '-images' + ssp[index],
            class_prob=True, certainty=True)
    
distr_rf_245 = rasterio.open("outputs/rf1-images245/probability_1.0.tif").read(1)
plotit(distr_rf_245, "NCR Range, averaged245", cmap="Greens")

distr_rf_126 = rasterio.open("outputs/rf1-images126/probability_1.0.tif").read(1)
plotit(distr_rf_126, "NCR Range, averaged126", cmap="Greens")

distr_rf_370 = rasterio.open("outputs/rf1-images370/probability_1.0.tif").read(1)
plotit(distr_rf_370, "NCR Range, averaged370", cmap="Greens")

distr_rf_585 = rasterio.open("outputs/rf1-images585/probability_1.0.tif").read(1)
plotit(distr_rf_585, "NCR Range, averaged585", cmap="Greens")

plotit(distr_rf_245[50:700, 100:1200], "smaller", cmap="Reds")

print('There are ', len(future_ras_feats), ' raster features.')
print(future_ras_feats)
# current order: 13, 14, 15, 2, 4, 5, 6

from pylab import plt
# define spatial plotter
def plotit(x, title, cmap="Blues"):
    plt.imshow(x, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title(title, fontweight = 'bold')

distr_rf = rasterio.open("outputs/rf1-images/probability_1.0.tif").read(1)
plotit(distr_rf, "NCR Range, averaged", cmap="Greens")

