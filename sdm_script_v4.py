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
import regionmask
import logging
import yaml
import sdm_func

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# logging
logging.basicConfig(
    filename=config['debug']['log_path'],
    filemode=config['debug']['write'],               # 'a' to append logs, 'w' to overwrite every run
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO          # Capture INFO, WARNING, ERROR, and CRITICAL logs
)



# THE ORDER IS SIGNIFICANT!!!!
ras_feats = glob.glob(config['current_climate']['bioclim_path'])

logging.info("Current raster features loaded: ".join(ras_feats))

# loading future prediction

future_ras_feats = []

for ssp_id, ssp_info in config['future_climate']['year_2070'].items():
    ssp_path = ssp_info["path"]
    out_subdir = ssp_info["trimmed_path"]
    os.makedirs(out_subdir, exist_ok=True)
        
    temp_future_ras_feats = []

    future_paths = glob.glob(ssp_path)
    logging.info(future_paths)
    for index, in_path in enumerate(future_paths):    
        with rasterio.open(in_path) as src:
            window = from_bounds(
                            config["future_climate"]['boundary']["min_lon"], config["future_climate"]['boundary']["min_lat"], config["future_climate"]['boundary']["max_lon"], config["future_climate"]['boundary']["max_lat"],
                            transform=src.transform
                        ).round_offsets().round_lengths()
            
            new_height = int(window.height * config["future_climate"]["downscale"])
            new_width = int(window.width * config["future_climate"]["downscale"])

            data = src.read(
                out_shape=(src.count, new_height, new_width), 
                window=window, resampling=Resampling.average
            )

            window_transform = src.window_transform(window)

            final_transform = window_transform * window_transform.scale(
                (window.width / new_width),
                (window.height / new_height))
    
            profile = src.profile.copy()
            # each file is single-band, and we're writing single-band outputs
            profile.update({
                "height": new_height,
                "width": new_width,
                "transform": final_transform,
                "count": 1
            })
            band = src.read(1, window=window)  # single-band tif => band 1

            

        out_path = os.path.join(out_subdir, f"trimmed_{config["current_climate"]["vars"][index]}")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(band, 1)
            dst.set_band_description(1, in_path)
        temp_future_ras_feats.append(out_path)

    future_ras_feats.append(temp_future_ras_feats)

window_lons = np.linspace(config["future_climate"]["boundary"]["min_lon"], config["future_climate"]["boundary"]["max_lon"], num=3540)
window_lats = np.linspace(config["future_climate"]["boundary"]["max_lat"], config["future_climate"]["boundary"]["min_lat"], num = 1785)


land_polygons = regionmask.defined_regions.natural_earth_v5_0_0.land_110
pred_mask = land_polygons.mask(window_lons, window_lats).values

land_mask = ~np.isnan(pred_mask) 

print("future ras feats loaded.")
logging.info("Future raster features loaded.")

for species_id, species_config in config['species'].items():
    species_name = species_config['name']
    species_path = species_config['dist_path']
    logging.info(species_name)
    try:
        subprocess.run([config['absence_gen']['rscript'], config['absence_gen']['pseudo_absence_path'], species_path, species_name], capture_output=True, 
            text=True, check = True)
    except subprocess.CalledProcessError as e:
        print(f"R Script failed with exit code {e.returncode}")
        print("Error Message:\n", e.stderr)
        logging.warning("R script failed. Proceeding if data files exist.")

    # ### Converting shapefiles (.shp) to geopandas dataframe

    ncr_gdf_mid = gpd.GeoDataFrame.from_file('outputs/data/' + species_name + '/mid.shp')
    # ncr_gdf_high = gpd.GeoDataFrame.from_file('data/' + species_name + '/high.shp')

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

    print("Original samples:", len(train_xs_mid))
    print("Clean samples:", len(train_xs_mid_clean))
    print("Dropped samples:", len(train_xs_mid) - len(train_xs_mid_clean))

    corr = x_df_mid.corr(method="pearson")
    print(corr)

    corr = np.abs(corr)
    plt.matshow(corr)
    plt.colorbar(label='Value Intensity')
    plt.savefig("outputs/correlations/correlation_matrix " + species_name)


    train_x_mid, test_x_mid, train_y_mid, test_y_mid = model_selection.train_test_split(train_xs_mid_clean, train_ys_mid_clean, test_size=0.25, random_state=42, stratify=train_ys_mid_clean)
    # train_xs_mid.shape, train_ys_mid.shape
    train_x_mid.shape, test_x_mid.shape, train_y_mid.shape, test_y_mid.shape

    print(train_x_mid)

    # Classifier implementation

    

    sample_weights_mid = compute_sample_weight(class_weight='balanced', y=train_y_mid)
    # sample_weights_high = compute_sample_weight(class_weight='balanced', y=train_y_high)
    k = 10
    kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

    rf = sdm_func.make_model_dict("random forest", RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1), "tree")

    lda = sdm_func.make_model_dict("lda", LinearDiscriminantAnalysis(), "linear")
    dtc = sdm_func.make_model_dict("decision tree", DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=42), "tree")
    lr = sdm_func.make_model_dict("log res glm", LogisticRegression(max_iter = 1000), "linear")
    lr1 = sdm_func.make_model_dict("log res big", LogisticRegression(max_iter=5000), "linear")
    xgb = sdm_func.make_model_dict("xg boost", XGBClassifier(), "tree")
    mars_model = make_pipeline(SplineTransformer(degree=1, n_knots=10, include_bias=False), LogisticRegression())
    mars = sdm_func.make_model_dict("mars", mars_model, "other")
    me = sdm_func.make_model_dict("maxent", MaxentModel(), "other")
    gam = sdm_func.make_model_dict("log gam", LogisticGAM(s(0) + s(1) + s(2), lam = 10), "linear")
    models = [rf, lda, dtc, lr, lr1, xgb, mars, me]

    
    cand_thresholds = np.linspace(0, 1, 1001)

    train_x_models, ensemble_val_x, train_y_models, ensemble_val_y = model_selection.train_test_split(train_x_mid, train_y_mid, test_size=0.2, random_state=42, stratify=train_y_mid)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model in models:
    # train the model (cross validate)
        thresholds = []
        msss_scores = []
        print(model["name"])
        logging.info("Model: " + model["name"])
        try:
            cv_score = model_selection.cross_val_score(model["model"], train_x_models, train_y_models, cv=skf)
            print("Cross validation score:", np.mean(cv_score))
            logging.info(model["name"] + " CV score: " + str(np.mean(cv_score)))
        except AttributeError as e:
            print("cross validation score not available for this model.")
            logging.warning(model["name"] + ": CV Score not available")

        for train_id, val_id in skf.split(train_x_models, train_y_models):
            x_train = train_x_models[train_id]
            x_val = train_x_models[val_id]
            y_train = train_y_models[train_id]
            y_val = train_y_models[val_id]

            model["model"].fit(x_train, y_train)
    # calculate probabilities with val data
            try:
                val_prob = model["model"].predict_proba(x_val)[:, 1]
            except IndexError as e:
                val_prob = model["model"].predict_proba(x_val)
    # scores = model_selection.cross_val_score(model["model"], train_x_mid, train_y_mid, cv=kf, scoring="")
    # calculate threshold

            best_threshold = 0
            best_sum = -1
            # best_sensit = -1
            # best_specif = -1

            for threshold in cand_thresholds:
                val_pred = (val_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

                sens_spec_sum = sensitivity + specificity

                if sens_spec_sum > best_sum:
                    best_sum = sens_spec_sum
                    best_threshold = threshold
                # best_sensitivity = sensitivity
                # best_specificity = specificity
        thresholds.append(best_threshold)
        msss_scores.append(best_sum)
        # model["threshold"] = best_threshold
        model["threshold"] = np.mean(thresholds)
        model["maxsss"] = np.mean(msss_scores)
        
        print("Best threshold:", model["threshold"])
        logging.info("Best threshold: " + str(model["threshold"]))

        print("Max sensitivity + specificity:", model["maxsss"])
        print("tss: ", model["maxsss"] - 1)
        logging.info("TSS: " + str(model["maxsss"] - 1))
    # retrain the model with all training data available
        model["model"].fit(train_x_models, train_y_models)
        model["frozen model"] = FrozenEstimator(model["model"])
        print("model frozen")

        try:
            test_prob = model["model"].predict_proba(test_x_mid)[:, 1]
        except IndexError as e:
            test_prob = model["model"].predict_proba(test_x_mid)

        test_pred = (test_prob >= model["threshold"]).astype(int)

    # find auc at the threshold
        auc_score = roc_auc_score(test_y_mid, test_pred)
        model["auc"] = auc_score
        print("AUC score: ", auc_score)
        logging.info("AUC score: " + str(auc_score))

        # try:
        #     results = permutation_importance(model["model"], test_x_mid, test_y_mid, scoring='roc_auc')
        #     for i, v in enumerate(results.importances_mean):
        #         print(f'Feature: {i}, Score: {v:.5f}')
        # except AttributeError as e:
        #     "permutation importance not available for this model"
        #     # plt.bar([x for x in range(len(results.importances_mean))], results.importances_mean)
        #     # plt.show()

        if model["class"] == "tree":
            importances = model["model"].feature_importances_
            feature_importance = pd.DataFrame({'Feature': x_df_mid.columns, 'Importance': importances})
            print(feature_importance.sort_values(by='Importance', ascending=False))
        elif model["class"] == "linear":
            importances = model["model"].coef_[0]
            feature_importance = pd.DataFrame({'Feature': x_df_mid.columns, 'Importance': importances})
            print(feature_importance.sort_values(by='Importance', ascending=False))

    # ensemble models: find the indices of the dictionary models with the highest auc and tss scores? 
    # find the highest auc and tss amongst different groups of classifiers: regressors, 
    

    auc_scores = []
    tss_scores = []

    for model in models:
        auc_scores.append(model["auc"])
        tss_scores.append(model["maxsss"] - 1)

    largest_auc = heapq.nlargest(3, range(len(auc_scores)), key=auc_scores.__getitem__)
    print(largest_auc)
    logging.info(largest_auc)

    # Loading future predictions

    # we use votingclassifier, but since we already have pretrained models, we use frozen classifiers as detailed here:
    # https://github.com/scikit-learn/scikit-learn/issues/12297
    
    ssp = ['126', '370', '585']

    future_target_xss = []
    future_raster_infos = []

    ensemble_models = [(models[x]["name"], models[x]["frozen model"]) for x in largest_auc]

    #impute cannot handle nans.............
    vc = VotingClassifier(ensemble_models, voting="soft")

    # all classifiers are frozen, so fitting has no effect, but is still a required action
    vc.fit(train_x_mid, train_y_mid)

    best_threshold = 0
    best_sum = -1
    # best_sensit = -1
    # best_specif = -1
    vc_val_prob = vc.predict_proba(ensemble_val_x)[:, 1]

    for threshold in cand_thresholds:
        vc_val_pred = (vc_val_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(ensemble_val_y, vc_val_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        sens_spec_sum = sensitivity + specificity

        if sens_spec_sum > best_sum:
            best_sum = sens_spec_sum
            best_threshold = threshold

    ensemble_test_prob = vc.predict_proba(test_x_mid)[:, 1]
    ensemble_test_pred = (ensemble_test_prob >= best_threshold).astype(int)
    ensemble_score = roc_auc_score(test_y_mid, ensemble_test_pred)
    print("Ensemble model score: ", ensemble_score)
    print("ensemble threshold: ", best_threshold, " ", best_sum)

    logging.info("Voting model ROC AUC: " + str(ensemble_score))
    logging.info("Voting model threshold: " + str(best_threshold))
    logging.info("Best sum: " + str(best_sum))


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
        presence_idx = 1
        prob_presence_valid = vc.predict_proba(future_target_xs_clean)[:, presence_idx]
        # prob_presence_valid = lr.predict_proba(future_target_xs_clean)[:, presence_idx]
        pred_valid = (prob_presence_valid >= best_threshold).astype("int16")
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
        masked_pred_to_write = np.where(land_mask, pred_to_write, np.nan)
        new_pred_to_write = np.where(np.isnan(masked_pred_to_write), -9999, pred_to_write)


        os.makedirs("outputs/predictions/" + species_name, exist_ok=True)
        with rasterio.open("outputs/predictions/" + species_name + "/prediction_" + species_name + "_" + ssp[index] + "_ensemble.tif", "w", **profile) as dst:
            dst.write(new_pred_to_write, 1)
            dst.set_band_description(1, "prediction")

    # distr_rf_pred_245 = rasterio.open("outputs/" + species_name + "/prediction_" + species_name + "_126" + "_ensemble.tif").read(1)
    # sdm_func.plotit(distr_rf_pred_245, species_name + ', 126', ["mediumseagreen", "orangered", "lightsteelblue"], cmap="Greens")

    # distr_rf_pred_245 = rasterio.open("outputs/" + species_name + "/prediction_" + species_name + "_370" + "_ensemble.tif").read(1)
    # sdm_func.plotit(distr_rf_pred_245, species_name + '370', ["mediumseagreen", "orangered", "lightsteelblue"], cmap="Greens")

    # distr_rf_pred_245 = rasterio.open("outputs/" + species_name + "/prediction_" + species_name + "_585" + "_ensemble.tif").read(1)
    # sdm_func.plotit(distr_rf_pred_245, species_name + '585', ["mediumseagreen", "orangered", "lightsteelblue"], cmap="Greens")
