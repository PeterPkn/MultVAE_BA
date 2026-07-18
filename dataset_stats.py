import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

def calculate_dataset_stats(base_dir="datasets/processed"):
    datasets = ["ml-1m", "lfm-demobias", "ekstrabladet"]
    FOLD = 4
    
    stats_list = []

    for dataset in datasets:
        fold_dir = os.path.join(base_dir, dataset, str(FOLD))
        
        if not os.path.exists(fold_dir):
            print(f"Skipping {dataset}: Fold {FOLD} data not found at {fold_dir}")
            continue

        try:
            train_input = sp.load_npz(os.path.join(fold_dir, "train_input.npz"))
            val_input = sp.load_npz(os.path.join(fold_dir, "val_input.npz"))
            val_target = sp.load_npz(os.path.join(fold_dir, "val_target.npz"))
            test_input = sp.load_npz(os.path.join(fold_dir, "test_input.npz"))
            test_target = sp.load_npz(os.path.join(fold_dir, "test_target.npz"))
        except FileNotFoundError as e:
            print(f"Missing some file for {dataset}: {e}")
            continue

        n_interactions = (
            train_input.nnz + 
            val_input.nnz + val_target.nnz + 
            test_input.nnz + test_target.nnz
        )

        n_users = train_input.shape[0] + val_input.shape[0] + test_input.shape[0]

        # works because: drop_fold_items=False
        n_items = train_input.shape[1]

        # calculate Sparsity
        sparsity = 1.0 - (n_interactions / (n_users * n_items))

        try:
            train_users = pd.read_csv(os.path.join(fold_dir, "train_user_info.csv"))
            val_users = pd.read_csv(os.path.join(fold_dir, "val_user_info.csv"))
            test_users = pd.read_csv(os.path.join(fold_dir, "test_user_info.csv"))
        except FileNotFoundError as e:
            print(f"Missing user info file for {dataset}: {e}")
            continue

        all_users = pd.concat([train_users, val_users, test_users], ignore_index=True)

        # Calculate Male/Female split
        if "gender" in all_users.columns:
            genders = all_users["gender"].astype(str).str.upper().str.strip()
            if genders.isin(["M", "F"]).all():
                n_male = (genders == "M").sum()
                n_female = (genders == "F").sum()
            else:
                n_male = (genders == "0.0").sum()
                n_female = (genders == "1.0").sum()
            gender_split = f"{n_male} M / {n_female} F"
            gender_ratio = n_male / n_female
        else:
            gender_split = "N/A"
            gender_ratio = np.nan
        stats_list.append({
            "Dataset": dataset,
            "Users": n_users,
            "Items": n_items,
            "Interactions": n_interactions,
            "Sparsity": f"{sparsity:.4%}",
            "M/F Split": gender_split,
            "M/F Ratio": f"{gender_ratio:.2f}"
        })

    # Print Results
    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        print("\n" + "="*80)
        print(f"{'DATASET STATISTICS (Calculated from Fold {})':^80}".format(FOLD))
        print("="*80)
        print(df_stats.to_string(index=False))
        print("="*80 + "\n")
    else:
        print("No valid datasets found to generate statistics.")

if __name__ == "__main__":
    calculate_dataset_stats()