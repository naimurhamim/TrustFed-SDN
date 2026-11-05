"""Data loading, cleaning, feature selection, partitioning, and scaling package."""

from trustfed.data.clean import (
    IDENTIFIER_AND_LEAKAGE_COLUMNS,
    CleaningStepRecord,
    clean_dataset,
    load_label_map,
    map_raw_label_to_canonical,
    save_cleaning_log,
)
from trustfed.data.load import (
    SUPPORTED_DATASETS,
    list_dataset_csv_files,
    load_raw_dataset,
    profile_dataframe,
    profile_raw_dataset,
    read_single_csv,
)
from trustfed.data.partition import (
    SplitBundle,
    compute_row_hashes,
    fit_and_apply_scalers,
    partition_dirichlet,
    partition_iid,
    partition_natural_non_iid,
    plot_partition_class_distributions,
    save_feature_selection_artifacts,
    select_topk_features_per_class,
    stratified_train_val_test_split,
    verify_zero_split_overlap,
)

__all__ = [
    "CleaningStepRecord",
    "IDENTIFIER_AND_LEAKAGE_COLUMNS",
    "SUPPORTED_DATASETS",
    "SplitBundle",
    "clean_dataset",
    "compute_row_hashes",
    "fit_and_apply_scalers",
    "list_dataset_csv_files",
    "load_label_map",
    "load_raw_dataset",
    "map_raw_label_to_canonical",
    "partition_dirichlet",
    "partition_iid",
    "partition_natural_non_iid",
    "plot_partition_class_distributions",
    "profile_dataframe",
    "profile_raw_dataset",
    "read_single_csv",
    "save_cleaning_log",
    "save_feature_selection_artifacts",
    "select_topk_features_per_class",
    "stratified_train_val_test_split",
    "verify_zero_split_overlap",
]
