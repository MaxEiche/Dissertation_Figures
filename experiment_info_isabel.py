# experiment_info.py
import os

# ─── Define all your experiments here ───
experiments = {

    "August_experiment_1x": {
        "experiment_path": r"C:\Users\ischnei\Desktop\Experiments\20250728_IsaS_twolayer_experiment_august\1x",
        "day_labels": [0, 2, 4, 7, 9, 11,14,16,18,21,23,25,28,30,32,35,37,39,42,44,46,49,51,53,56,58,60,63,65,67,70,72],
        "calibration_factor": 0.46100721548394352732781071743631,
        "treatment_excel": r"C:\Users\ischnei\Desktop\Experiments\20250728_IsaS_twolayer_experiment_august\1x\timestacks\total_fluorescence_intensity.xlsx",
        "treatment_excel_sheet_name": "Total_normalized",
        "excel_mode": " timepoints_horizontal_replicates",
        "THRESHOLD_LOW": 99.5,
        "THRESHOLD_HIGH": 149.5,
        "BG_EGFP": 7343,
        "BG_Calibrated_mScarlet":  2961,
        "BG_Total": 10304,

    },    # …add as many as you like
}

def get_config(name):
    cfg = experiments[name]
    exp_path = cfg["experiment_path"]
    return {
        "experiment_path": exp_path,
        "day_labels": cfg["day_labels"],
        "tifs_dir": os.path.join(exp_path, "Tifs"),
        "timestacks_dir": os.path.join(exp_path, "timestacks"),
        "videos_dir": os.path.join(exp_path, "videos"),
        "metadata_path": os.path.join(exp_path, "timestacks", "metadata.xlsx"),
        "resistance_metadata_path": os.path.join(exp_path, "resistance_group", "metadata.xlsx"),
        "scaling_factor": 1.0,
        "channels": [2, 0, 1],
        "ch_map": {"mScarlet": "res", "EGFP": "sen"},
        "excel_additional_dir": os.path.join(exp_path, "timestacks", "Excel", "Additional_info"),
        "all_data_nested_pkl": os.path.join(exp_path, "timestacks", "all_data_nested.pkl"),
        "all_fluorescence_data_pkl": os.path.join(exp_path, "timestacks", "all_fluorescence_data.pkl"),
        "all_data_total_norm.pkl": os.path.join(exp_path, "timestacks", "all_data_total_norm.pkl"),
        "fluorescence_excel": os.path.join(exp_path, "timestacks", "Fluorescence_intensity_data.xlsx"),
        "normalized_replicates_excel": os.path.join(exp_path, "timestacks", "Normalized_replicates.xlsx"),
        "tf_total_average_excel": os.path.join(exp_path, "timestacks", "TFluorescence_intensity_data_total_average.xlsx"),
        "total_fluorescence_excel": os.path.join(exp_path, "timestacks", "total_fluorescence_intensity.xlsx"),
        "derivative_excel": os.path.join(exp_path, "timestacks", "total_fluorescence_intensities_and_derivatives_BGsub.xlsx"), #BG subtracted!
        "calibration_factor": cfg["calibration_factor"],

        #BG values (default to 0 if missing for an experiment)
        "BG_EGFP": cfg.get("BG_EGFP", 0.0),
        "BG_Calibrated_mScarlet": cfg.get("BG_Calibrated_mScarlet", 0.0),
        "BG_Total": cfg.get("BG_Total", 0.0),
    }
