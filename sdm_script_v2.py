# # SDM in Python
# 
# Some features based on https://github.com/daniel-furman/eco-distribution-mapping/blob/main/Python-sdm.ipynb

# ### Generate Pseudo-Absence Data
# Absence data must be generated for these classifiers, and different quantities must be generated for types of classifiers, based on Barbet-Massin et al., 2012.

# ### Making subdirectories
# inputs - files input for processing   
# outputs - results from processing

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
import rasterio
from pyimpute import load_training_vector
from pyimpute import load_targets
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


files_list = ["nst_pest_sightings\\bean_leaf_beetle_0003221-260623161305970\\0003221-260623161305970.csv", 
              "nst_pest_sightings\\bird_cherry_aphid_0003207-260623161305970\\0003207-260623161305970.csv",
              "nst_pest_sightings\\black_cutworm_0003212-260623161305970\\0003212-260623161305970.csv",
            #   "nst_pest_sightings\\corn_leaf_aphid_0003215-260623161305970\\0003215-260623161305970.csv",
              "nst_pest_sightings\\differential_grasshopper0032387-260623161305970\\0032387-260623161305970.csv",
            #   "nst_pest_sightings\\european_corn_borer0003385-260623161305970\\0003385-260623161305970.csv",
              "nst_pest_sightings\\green_stink_0005409-260623161305970\\0005409-260623161305970.csv",
            #   "nst_pest_sightings\\hessian_fly_0003394-260623161305970\\0003394-260623161305970.csv",
              "nst_pest_sightings\\japanese_beetle_0003383-260623161305970\\0003383-260623161305970.csv",
              "nst_pest_sightings\\northern_corn_rootworm_0003360-260623161305970\\0003360-260623161305970.csv",
            #   "nst_pest_sightings\\reg_legged_grasshopperobservations-752840.csv\\observations-752840.csv",
              "nst_pest_sightings\\seedcorn_maggot_0003200-260623161305970\\0003200-260623161305970.csv",
              "nst_pest_sightings\\southern_green_stink_0003348-260623161305970\\0003348-260623161305970.csv",
              "nst_pest_sightings\\three_cornered_alfalfa_0003390-260623161305970\\0003390-260623161305970.csv",
              "nst_pest_sightings\true_armyworm0032387-260623161305970\\0032387-260623161305970.csv",
              "nst_pest_sightings\\two_striped_grasshopper0032397-260623161305970\\0032397-260623161305970.csv",
              "nst_pest_sightings\\western_corn_rootworm_0003372-260623161305970\\0003372-260623161305970.csv"
              ]

# for file in files_list:
#     with open(file, mode='r', encoding='utf-8') as file:
#         row_count = sum(1 for line in file)

#     print(f"{file}, Total lines: {row_count}")

# run the subprocesses in R
command_blb = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[0], "northerncorn"]

command_bca = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[1], "birdcherryaphid"]

command_bc = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[2], "blackcutworm"]

command_dg = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[3], "differentialgrasshopper"]

command_gs = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[4], "greenStink"]

command_jb = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[5], "japanesebeetle"]

command_ncr = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[6], "northerncornrootworm"]

command_sm = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[7], "seedcornmaggot"]

command_sgs = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[8], "southerngreenstink"]

command_tca = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[9], "threecornerneredalfalfa"]

command_ta = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[10], "truearmyworm"]

command_tsg = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[11], "truearmyworm"]

command_wcr = ["C:\\Users\\chenst-intern\\AppData\\Local\\Programs\\R\\R-4.6.1\\bin\\x64\\Rscript.exe",'pseudo_absence.R', 
           files_list[12], "westcornrootworm"]

command_list = [command_blb, command_bca, command_bc, command_dg, command_gs, command_jb, command_ncr, command_sm, command_sgs, command_tca, command_ta, command_tsg, command_wcr]


# load climate data
ras_feats = sorted(glob.glob(
    'worldclim/wc2.1_2.5m_bio_*.tif'))

# raster_crs is a global variable
with rasterio.open(ras_feats[0]) as src:
        raster_crs = src.crs

# THE ORDER IS SIGNIFICANT!!!!
print('There are ', len(ras_feats), ' raster features.')
print(ras_feats)
# current order: 13, 14, 15, 2, 4, 5, 6

for command in command_list:    
    try:
        subprocess.run(command, capture_output=True, 
            text=True, check = True)
    except subprocess.CalledProcessError as e:
        print(f"R Script failed with exit code {e.returncode}")
        print("Error Message:\n", e.stderr)


# ### Converting shapefiles (.shp) to geopandas dataframe
    ncr_gdf_mid = gpd.GeoDataFrame.from_file('data/' + command[3] + '/mid.shp')
    ncr_gdf_high = gpd.GeoDataFrame.from_file('data/' + command[3] + '/high.shp')
# for f in sorted(glob.glob('data/ncr*')):
#     shutil.copy(f, 'inputs/')
# ncr_gdf_mid.sample(5)
# Checking duplicates and NA values. Coordinate reference system should ideally be epsg: 4326
    print("Number of duplicates: ", ncr_gdf_mid.duplicated(subset='geometry', keep='first').sum())
    print("Number of NA's: ", ncr_gdf_mid['geometry'].isna().sum())
    print("Coordinate reference system: {}".format(ncr_gdf_mid.crs))
    print("{} observations with {} columns".format(*ncr_gdf_mid.shape))

    print("Number of duplicates: ", ncr_gdf_high.duplicated(subset='geometry', keep='first').sum())
    print("Number of NA's: ", ncr_gdf_high['geometry'].isna().sum())
    print("Coordinate reference system: {}".format(ncr_gdf_high.crs))
    print("{} observations with {} columns".format(*ncr_gdf_high.shape))
# Mapping the species presences (pa == 1)
# Mapping the species absences (pa == 0)

    ncr_gdf_mid[ncr_gdf_mid.CLASS == 1].plot(marker='*', color='green', markersize=5)
    ncr_gdf_mid[ncr_gdf_mid.CLASS == 0].plot(marker='*', color='red', markersize=5)
# ### Classifier Training

# TO MOVE FILES INTO INPUT
# for f in sorted(glob.glob('worldclim/wc2.1_2.5m_bio_*.tif')):
#     shutil.copy(f,'inputs/')
    ncr_gdf_mid = ncr_gdf_mid.to_crs(raster_crs)
    ncr_gdf_high = ncr_gdf_high.to_crs(raster_crs)

    ncr_gdf_low = []
    for i in range(1,11):
        ncr_gdf_low.append(gpd.GeoDataFrame.from_file('data/' + command[3] + '/low/'+ str(i) + '.shp').to_crs(raster_crs))

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

    print("Original samples:", len(train_xs_mid))
    print("Clean samples:", len(train_xs_mid_clean))
    print("Dropped samples:", len(train_xs_mid) - len(train_xs_mid_clean))

    train_x_mid, test_x_mid, train_y_mid, test_y_mid = model_selection.train_test_split(train_xs_mid_clean, train_ys_mid_clean, test_size=0.25, random_state=42, stratify=train_ys_mid_clean)

    train_xs_high, train_ys_high = load_training_vector(ncr_gdf_high, ras_feats, response_field='CLASS')

    x_df_high = pd.DataFrame(train_xs_high)
    x_df_high = x_df_high.apply(pd.to_numeric, errors="coerce")
    x_df_high = x_df_high.replace([np.inf, -np.inf], np.nan)

    y_ser_high = pd.Series(train_ys_high)
    mask_high = x_df_high.notna().all(axis=1) & y_ser_high.notna()

    # Apply mask_mid to both X and y
    train_xs_high_clean = x_df_high.loc[mask_high].to_numpy(dtype=float)
    train_ys_high_clean = y_ser_high.loc[mask_high].to_numpy()

    print("Original samples:", len(train_xs_high))
    print("Clean samples:", len(train_xs_high_clean))
    print("Dropped samples:", len(train_xs_high) - len(train_xs_high_clean))

    train_x_high, test_x_high, train_y_high, test_y_high = model_selection.train_test_split(train_xs_high_clean, train_ys_high_clean, test_size=0.25, random_state=42, stratify=train_ys_high_clean)
# train_xs_mid.shape, train_ys_mid.shape
    train_x_high.shape, test_x_high.shape, train_y_high.shape, test_y_high.shape
# Classifier implementation

    sample_weights_mid = compute_sample_weight(class_weight='balanced', y=train_y_mid)
    sample_weights_high = compute_sample_weight(class_weight='balanced', y=train_y_high)
    k = 4
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

### random forest ###
    auc_scores = []
    acc_scores = []
    logloss_scores = []

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1)
    rf_accuracy_scores = model_selection.cross_val_score(rf, train_x_mid, train_y_mid, cv=kf, scoring='balanced_accuracy')
    print("rf %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, rf_accuracy_scores.mean() * 100, rf_accuracy_scores.std() * 200))
    rf.fit(train_x_mid, train_y_mid)
    rf_train_pred = rf.predict(train_x_mid)

    rf_test_prob = rf.predict_proba(test_x_mid)[:, 1]
    rf_test_pred = rf.predict(test_x_mid)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, rf_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, rf_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, rf_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, rf_test_pred))
    logloss_scores.append(log_loss(test_y_mid, rf_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

### LDA ###

    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    lda = LinearDiscriminantAnalysis()
    lda_accuracy_scores = model_selection.cross_val_score(lda, train_x_mid, train_y_mid, cv = kf, scoring='balanced_accuracy')
    print("lda %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, lda_accuracy_scores.mean() * 100, lda_accuracy_scores.std() * 200))
    lda.fit(train_x_mid, train_y_mid)
    lda_train_pred = lda.predict(train_x_mid)
    lda_test_pred = lda.predict(test_x_mid)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, lda_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, lda_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, lda_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, lda_test_pred))
    logloss_scores.append(log_loss(test_y_mid, lda_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

### decision tree classifier ###
    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    dtc = DecisionTreeClassifier()
    dtc_accuracy_scores = model_selection.cross_val_score(dtc, train_x_mid, train_y_mid, cv = kf, scoring='balanced_accuracy')
    print("dtc %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, dtc_accuracy_scores.mean() * 100, dtc_accuracy_scores.std() * 200))
    dtc.fit(train_x_mid, train_y_mid)
    dtc_train_pred = dtc.predict(train_x_mid)
    dtc_test_pred = dtc.predict(test_x_mid)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, dtc_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, dtc_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, dtc_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, dtc_test_pred))
    logloss_scores.append(log_loss(test_y_mid, dtc_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

    # ### logistic regression (GLM) ###
    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    lr = LogisticRegression(max_iter=5000)
    lr_accuracy_scores = model_selection.cross_val_score(lr, train_x_mid, train_y_mid, cv = kf, scoring='balanced_accuracy')
    print("lr %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, lr_accuracy_scores.mean() * 100, lr_accuracy_scores.std() * 200))
    lr.fit(train_x_mid, train_y_mid)
    lr_train_pred = lr.predict(train_x_mid)
    lr_test_pred = lr.predict(test_x_mid)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, lr_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, lr_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, lr_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, lr_test_pred))
    logloss_scores.append(log_loss(test_y_mid, lr_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))


    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    lr1 = LogisticRegression(max_iter=5000)
    lr1_accuracy_scores = model_selection.cross_val_score(lr1, train_x_high, train_y_high, cv = kf, scoring='balanced_accuracy')
    print("lr1 %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, lr1_accuracy_scores.mean() * 100, lr1_accuracy_scores.std() * 200))
    lr1.fit(train_x_high, train_y_high)
    lr1_train_pred = lr1.predict(train_x_high)
    lr1_test_pred = lr1.predict(test_x_high)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_high, lr1_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_high, lr1_test_pred))

    auc_scores.append(roc_auc_score(test_y_high, lr1_test_pred))
    acc_scores.append(accuracy_score(test_y_high, lr1_test_pred))
    logloss_scores.append(log_loss(test_y_high, lr1_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

    ### xgb classifier ###
    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    xgb = XGBClassifier()
    xgb_accuracy_scores = model_selection.cross_val_score(xgb, train_x_mid, train_y_mid, cv = kf, scoring='balanced_accuracy')
    print("xgb %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, xgb_accuracy_scores.mean() * 100, xgb_accuracy_scores.std() * 200))
    xgb.fit(train_x_mid, train_y_mid)
    xgb_train_pred = xgb.predict(train_x_mid)
    xgb_test_pred = xgb.predict(test_x_mid)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, xgb_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, xgb_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, xgb_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, xgb_test_pred))
    logloss_scores.append(log_loss(test_y_mid, xgb_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

    ### light gbm ###
    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    lgbm = LGBMClassifier()
    lgbm_accuracy_scores = model_selection.cross_val_score(lgbm, train_x_mid, train_y_mid, cv = kf, scoring='balanced_accuracy')
    print("lgbm %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, lgbm_accuracy_scores.mean() * 100, lgbm_accuracy_scores.std() * 200))
    lgbm.fit(train_x_mid, train_y_mid)
    lgbm_train_pred = lgbm.predict(train_x_mid)
    lgbm_test_pred = lgbm.predict(test_x_mid)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, lgbm_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, lgbm_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, lgbm_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, lgbm_test_pred))
    logloss_scores.append(log_loss(test_y_mid, lgbm_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

    ## MARS ###
    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    mars = make_pipeline(SplineTransformer(degree=1, n_knots=10, include_bias=False), LogisticRegression())
    mars_accuracy_scores = model_selection.cross_val_score(mars, train_x_mid, train_y_mid, cv = kf, scoring='balanced_accuracy')
    print("mars %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, mars_accuracy_scores.mean() * 100, mars_accuracy_scores.std() * 200))
    mars.fit(train_x_mid, train_y_mid)
    mars_train_pred = mars.predict(train_x_mid)
    mars_test_pred = mars.predict(test_x_mid)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, mars_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, mars_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, mars_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, mars_test_pred))
    logloss_scores.append(log_loss(test_y_mid, mars_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

    ### MAXENT ###
    auc_scores.clear()
    acc_scores.clear()
    logloss_scores.clear()

    maxent = MaxentModel()
    maxent_accuracy_scores = model_selection.cross_val_score(maxent, train_x_mid, train_y_mid, cv = kf, scoring='roc_auc')
    print("maxent %d-fold Cross Validation Accuracy: %0.2f (+/- %0.2f)"
            % (k, maxent_accuracy_scores.mean() * 100, maxent_accuracy_scores.std() * 200))
    maxent.fit(train_x_mid, train_y_mid)
    maxent_train_scores = maxent.predict(train_x_mid)
    maxent_test_scores = maxent.predict(test_x_mid)

    maxent_train_pred = (maxent_train_scores >= 0.5).astype(int)
    maxent_test_pred = (maxent_test_scores >= 0.5).astype(int)

    print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, maxent_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, maxent_test_pred))

    auc_scores.append(roc_auc_score(test_y_mid, maxent_test_pred))
    acc_scores.append(accuracy_score(test_y_mid, maxent_test_pred))
    logloss_scores.append(log_loss(test_y_mid, maxent_test_pred))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))




    # # 'et': (ExtraTreesClassifier()),
    # 'lda': (LinearDiscriminantAnalysis()),
    # 'dtc': (DecisionTreeClassifier()),
    # 'glm': (LogisticRegression()),
    # 'xgb': (XGBClassifier()),
    # 'lgbm': (LGBMClassifier()),
    # # 'maxent': (MaxentModel())



# for name, (model) in CLASS_MAP.items():
#      # k-fold
#     # spatial prediction
#     score = model.score(test_x_mid, test_y_mid)
#     print(name + " score: %f" % (score))
    # impute(target_xs, model, raster_info, outdir='outputs/' + name + '-images',
    #        class_prob=True, certainty=True)


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    acc_scores = []
    logloss_scores = []

    gam = LogisticGAM(s(0) + s(1) + s(2))

    # for train_idx, test_idx in cv.split(train_x_mid, train_y_mid):
    #     X_train, X_test = train_x_mid[train_idx], train_x_mid[test_idx]
    #     y_train, y_test = train_y_mid[train_idx], train_y_mid[test_idx]
    #     gam.fit(X_train, y_train)
    for train_idx, test_idx in cv.split(train_x_high, train_y_high):
        X_train, X_test = train_x_high[train_idx], train_x_high[test_idx]
        y_train, y_test = train_y_high[train_idx], train_y_high[test_idx]
        gam.fit(X_train, y_train)

        y_prob = gam.predict_proba(X_test)

        y_pred = gam.predict(X_test)

        auc_scores.append(roc_auc_score(y_test, y_prob))
        acc_scores.append(accuracy_score(y_test, y_pred))
        logloss_scores.append(log_loss(y_test, y_prob))

    print("Mean AUC:", np.mean(auc_scores))
    print("Mean accuracy:", np.mean(acc_scores))
    print("Mean log loss:", np.mean(logloss_scores))

    # gam_train_pred = gam.predict(train_x_mid)
    # gam_test_pred = gam.predict(test_x_mid)
    # print("Train balanced accuracy:", balanced_accuracy_score(train_y_mid, gam_train_pred))
    # print("Test balanced accuracy:", balanced_accuracy_score(test_y_mid, gam_test_pred))

    gam_train_pred = gam.predict(train_x_high)
    gam_test_pred = gam.predict(test_x_high)
    print("Train balanced accuracy:", balanced_accuracy_score(train_y_high, gam_train_pred))
    print("Test balanced accuracy:", balanced_accuracy_score(test_y_high, gam_test_pred))

# Confusion Matrix, for calculating MaxSSS (maximum sum of sensitivity and specificity), also related to Youden's J statistic
# spoiler: its iterative


    thresholds = np.linspace(0, 1, 1001)

    best_threshold = 0
    best_sum = -1
    best_sensit = -1
    best_specif = -1

##### STOPPED HEREE ######## 7/14 17:000

    for threshold in thresholds:
        y_pred = (rf_test_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_y_mid, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan


        sens_spec_sum = sensitivity + specificity

        if sens_spec_sum > best_sum:
            best_sum = sens_spec_sum
            best_threshold = threshold
            best_sensitivity = sensitivity
            best_specificity = specificity

print("Best threshold:", best_threshold)
print("Max sensitivity + specificity:", best_sum)
print("Sensitivity:", best_sensitivity)
print("Specificity:", best_specificity)
print("Youden J:", best_sum - 1)



# %% [markdown]
# Averaging the outputs of the models, and plotting it onto a map

# %%
# NOT LORE ACCURATE ANYMORE
# from pylab import plt
# # define spatial plotter
# def plotit(x, title, cmap="Blues"):
#     plt.imshow(x, cmap=cmap, interpolation='nearest')
#     plt.colorbar()
#     plt.title(title, fontweight = 'bold')

# distr_rf = rasterio.open("outputs/rf-images/probability_1.0.tif").read(1)
# distr_et = rasterio.open("outputs/et-images/probability_1.0.tif").read(1)
# distr_xgb =  rasterio.open("outputs/xgb-images/probability_1.tif").read(1)
# distr_lgbm =  rasterio.open("outputs/lgbm-images/probability_1.0.tif").read(1)
# distr_averaged = (distr_rf + distr_et + distr_xgb + distr_lgbm)/4  

# plotit(distr_averaged, "NCR Range, averaged", cmap="Greens")

# %%
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
        ).round_offsets().round_lengths()

        data = src.read(1, window=window)

        new_transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": new_transform,
            "count": 1
        })
        
        temp_future_ras_feats = []
        for feature_name in training_feature_names:
            
            band_index = future_band_map[feature_name]

            band = src.read(band_index, window=window)

            out_path = os.path.join(f"{out_dir}_{index}", f"future_{feature_name}.tif")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(band, 1)
                dst.set_band_description(1, feature_name)

            temp_future_ras_feats.append(out_path)

    future_ras_feats.append(temp_future_ras_feats)
    # future_ras_feats = sorted(glob.glob(
    #     'future_split_bands_1/future_bio*.tif')) 



# %%
print(future_ras_feats)

# %%
ssp = ['126', '245', '370', '585']

future_target_xss = []
future_raster_infos = []
for index, feat in enumerate(future_ras_feats):
    future_target_xs, future_raster_info = load_targets(feat)
    print(future_raster_info)

    future_target_xs_df = pd.DataFrame(future_target_xs)
    future_target_xs_df.sample(5)
    future_target_xs_df = future_target_xs_df.apply(pd.to_numeric, errors="coerce")
    future_target_xs_df = future_target_xs_df.replace([np.inf, -np.inf], np.nan)


    mask_future = future_target_xs_df.notna().all(axis=1)

# Apply mask_mid to both X and y
    future_target_xs_clean = future_target_xs_df.loc[mask_future].to_numpy(dtype=float)

    # print("X_valid shape:", future_target_xs_clean.shape)

    # pred_valid = rf.predict(future_target_xs_clean)
    # print("Prediction classes/counts:", np.unique(pred_valid, return_counts=True))

#impute cannot handle nans.............
    presence_idx = 1
    prob_presence_valid = lr.predict_proba(future_target_xs_clean)[:, presence_idx]
    threshold = 0.3
    pred_valid = (prob_presence_valid >= threshold).astype("int16")
    # pred_valid = rf.predict(future_target_xs_clean)

    n_pixels = future_target_xs_df.shape[0]
    pred_full = np.full(n_pixels, np.nan)
    pred_full[mask_future.to_numpy()] = pred_valid

    height = future_raster_info["shape"][0]
    width = future_raster_info["shape"][1]

    pred_raster = pred_full.reshape(height, width)
    # impute(future_target_xs_clean, rf, future_raster_info, outdir='outputs/' + 'rf1' + '-images' + ssp[index],
    #         class_prob=True, certainty=True)
    
    # impute(future_target_xs_clean, lr, future_raster_info, outdir='outputs/' + 'lr' + '-images' + ssp[index],
    #         class_prob=True, certainty=True)
    
    # impute(future_target_xs_clean, lr1, future_raster_info, outdir='outputs/' + 'lr1' + '-images' + ssp[index],
            # class_prob=True, certainty=True)

    # profile = future_raster_info["profile"].copy()

    profile.update({
        "driver": "GTiff",
        "height": pred_raster.shape[0],
        "width": pred_raster.shape[1],
        "count": 1,
        "dtype": "float32",
        "nodata": -9999
    })

    pred_to_write = pred_raster.astype("float32")
    pred_to_write = np.where(np.isnan(pred_to_write), -9999, pred_to_write)

    with rasterio.open("outputs/prediction" + ssp[index] + "lr.tif", "w", **profile) as dst:
        dst.write(pred_to_write, 1)
        dst.set_band_description(1, "prediction")

# %%
distr_rf_pred_245 = rasterio.open("outputs/prediction126.tif").read(1)
plotit(distr_rf_pred_245, "NCR Range, averaged245", cmap="Greens")

# %%


# %%
distr_rf_245 = rasterio.open("outputs/rf1-images245/probability_1.0.tif").read(1)
plotit(distr_rf_245, "NCR Range, averaged245", cmap="Greens")



# %%
distr

# %%
distr_rf_126 = rasterio.open("outputs/rf1-images126/probability_1.0.tif").read(1)
plotit(distr_rf_126, "NCR Range, averaged126", cmap="Greens")

# %%
distr_rf_370 = rasterio.open("outputs/rf1-images370/probability_1.0.tif").read(1)
plotit(distr_rf_370, "NCR Range, averaged370", cmap="Greens")

# %%
distr_rf_585 = rasterio.open("outputs/rf1-images585/probability_1.0.tif").read(1)
plotit(distr_rf_585, "NCR Range, averaged585", cmap="Greens")

# %%
plotit(distr_rf_585[50:700, 100:1200], "smaller", cmap="Reds")

# %%
plotit(distr_rf_126[50:700, 100:1200], "smaller", cmap="Reds")

# %%
# THE ORDER IS SIGNIFICANT!!!!

print('There are ', len(future_ras_feats), ' raster features.')
print(future_ras_feats)
# current order: 13, 14, 15, 2, 4, 5, 6

# %%


# %%
from pylab import plt
# define spatial plotter
def plotit(x, title, cmap="Blues"):
    plt.imshow(x, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title(title, fontweight = 'bold')

distr_rf = rasterio.open("outputs/rf1-images/probability_1.0.tif").read(1)
plotit(distr_rf, "NCR Range, averaged", cmap="Greens")


