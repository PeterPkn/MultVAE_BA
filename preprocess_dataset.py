# %%
import os
import numpy as np
import pandas as pd
from enum import Enum
from scipy import sparse as sp
from sklearn.model_selection import KFold, train_test_split
import json
from ruamel.yaml import YAML

# UTILS

yaml = YAML(typ="safe")

directory_char = "/" if os.name != "nt" else "\\"

def json_load(file_path):
    with open(file_path, "r") as fh:
        return json.load(fh)

def json_dump(obj, file_path):
    with open(file_path, "w") as fh:
        return json.dump(obj, fh, indent=4)

def yaml_load(file_path):
    with open(file_path, "r") as fh:
        return yaml.load(fh)

def yaml_dump(obj, file_path):
    with open(file_path, "w") as fh:
        return yaml.dump(obj, fh)


def split_interactions(interaction_matrix, test_size=0.2, random_state=42):
    user_idxs, item_idxs, _ = sp.find(interaction_matrix == 1)

    tr_ind, te_ind = train_test_split(
        np.arange(len(user_idxs)), test_size=test_size, random_state=random_state
    )

    tr_values = np.ones(len(tr_ind))
    tr_matrix = sp.csr_matrix(
        (tr_values, (user_idxs[tr_ind], item_idxs[tr_ind])),
        shape=interaction_matrix.shape,
    )

    te_values = np.ones(len(te_ind))
    te_matrix = sp.csr_matrix(
        (te_values, (user_idxs[te_ind], item_idxs[te_ind])),
        shape=interaction_matrix.shape,
    )

    return tr_matrix, te_matrix


def split_user_interactions(interaction_matrix, test_size=0.2, random_state=42):
    tr_user, te_user, tr_items, te_items = [], [], [], []

    for user, _ in enumerate(interaction_matrix):
        user_idx, item_idx, _ = sp.find(interaction_matrix[user] == 1)
        if len(item_idx) >= 5:
            train_items, test_items = train_test_split(
                item_idx, test_size=test_size, random_state=random_state
            )
            tr_user += [user] * len(train_items)
            te_user += [user] * len(test_items)
            tr_items = np.concatenate((tr_items, train_items), axis=0)
            te_items = np.concatenate((te_items, test_items), axis=0)
    tr_values = np.ones(len(tr_user))
    tr_matrix = sp.csr_matrix(
        (tr_values, (tr_user, tr_items)),
        shape=interaction_matrix.shape,
    )
    te_values = np.ones(len(te_user))
    te_matrix = sp.csr_matrix(
        (te_values, (te_user, te_items)),
        shape=interaction_matrix.shape,
    )

    return tr_matrix, te_matrix


def drop_users_no_inter(
    interaction_matrix_train, interaction_matrix_test, df_user_info
):
    """
    For datasets where users have only very few interactions with items, it may happen that
    a user has no interactions in a specific split. We mitigate this by filtering out such users.
    """
    zero_mask = (interaction_matrix_train.sum(axis=1) == 0) | (
        interaction_matrix_test.sum(axis=1) == 0
    )
    zero_mask = np.array(zero_mask).flatten()
    interaction_matrix_train = interaction_matrix_train[~zero_mask, :]
    interaction_matrix_test = interaction_matrix_test[~zero_mask, :]

    df_user_info = df_user_info.loc[~zero_mask, :]
    df_user_info.reset_index(drop=True, inplace=True)
    df_user_info = df_user_info.assign(userID=df_user_info.index)
    return interaction_matrix_train, interaction_matrix_test, df_user_info


def select_and_reset(df_user_info, indices):
    df_user_info = df_user_info.iloc[indices].copy()
    df_user_info.reset_index(drop=True, inplace=True)
    df_mapping = pd.DataFrame.from_dict(
        {"old": df_user_info["userID"].copy(), "new": df_user_info.index}
    )
    df_user_info["userID"] = df_user_info.index
    return df_user_info, df_mapping


def split_and_store(
    split_indices,
    interaction_matrix,
    df_user_info,
    random_state,
    storage_dir,
    split_abbrev,
):
    """
    Performs the data preparation for validation or test data. Results will then be stores.
    """
    im = interaction_matrix[split_indices, :]
    tr_im, te_im = split_user_interactions(im, random_state=random_state)

    df_user_info, df_mapping = select_and_reset(df_user_info, split_indices)
    tr_im, te_im, df_user_info = drop_users_no_inter(tr_im, te_im, df_user_info)

    df_user_info.to_csv(
        os.path.join(storage_dir, f"{split_abbrev}_user_info.csv"), index=False
    )
    df_mapping.to_csv(
        os.path.join(storage_dir, f"{split_abbrev}_user_mapping.csv"), index=False
    )

    sp.save_npz(os.path.join(storage_dir, f"{split_abbrev}_input.npz"), tr_im)
    sp.save_npz(os.path.join(storage_dir, f"{split_abbrev}_target.npz"), te_im)


def perform_kfold_split(
    n_folds: int,
    interaction_matrix,
    df_user_info: pd.DataFrame,
    storage_dir: str,
    features: list,
    random_state: int = 42,
    drop_fold_items=True,
):
    print(f"Creating {n_folds} folds for cross validation")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # generate splits
    X = np.arange(interaction_matrix.shape[0])
    fold_indices = [indices for _, indices in kf.split(X)]

    # Split data into different folds and store them to reduce compute time while training
    for i in range(n_folds):
        print("Splitting data for fold", i)
        fold_interaction_matrix = interaction_matrix.copy()

        fold_dir = os.path.join(storage_dir, str(i))
        os.makedirs(fold_dir, exist_ok=True)

        n_tr_folds = n_folds - 2
        tr_indices = np.concatenate(
            [fold_indices[i] for i in (i + np.arange(n_tr_folds)) % n_folds]
        )
        tr_indices = np.sort(tr_indices)

        tr_im = fold_interaction_matrix[tr_indices, :]
        tr_user_info, df_mapping = select_and_reset(df_user_info, tr_indices)
        if drop_fold_items:
            # Ensure that all items in the training data feature user interactions
            zero_mask = np.array(tr_im.sum(axis=0) == 0).flatten()
            dropped_item_indices = np.argwhere(zero_mask).flatten().tolist()
            yaml_dump(
                {"dropped_item_indices": dropped_item_indices},
                os.path.join(fold_dir, "dropped_items.yaml"),
            )

            tr_im = tr_im[:, ~zero_mask]
            fold_interaction_matrix = fold_interaction_matrix[:, ~zero_mask]

        tr_user_info.to_csv(os.path.join(fold_dir, "train_user_info.csv"), index=False)
        sp.save_npz(os.path.join(fold_dir, "train_input.npz"), tr_im)

        # ===== Validation data ======
        vd_indices = np.sort(fold_indices[(n_tr_folds + i) % n_folds])
        split_and_store(
            vd_indices,
            fold_interaction_matrix,
            df_user_info,
            random_state,
            fold_dir,
            split_abbrev="val",
        )

        # ===== Test data ======
        te_indices = np.sort(fold_indices[(n_tr_folds + i + 1) % n_folds])
        split_and_store(
            te_indices,
            fold_interaction_matrix,
            df_user_info,
            random_state,
            fold_dir,
            split_abbrev="test",
        )

        # ===== Validating the results =====
        # Ensure that no indices overlap between the different data sets
        n_indices_total = len(tr_indices) + len(vd_indices) + len(te_indices)
        all_indices = np.union1d(np.union1d(tr_indices, vd_indices), te_indices)
        assert n_indices_total == len(
            all_indices
        ), f"User indices of different splits overlap in fold {i}"


def ensure_make_data(
    data_dir: str,
    n_folds: int,
    target_path: str,
    features: list = None,
    random_state: int = 42,
    drop_fold_items=True,
):
    """
    Ensure that dataset is prepared for experiments by performing `n_folds` fold splitting for cross-validation and
    possible resampling.

    @data_dir: the path to the dataset
    @n_folds: the number of folds for k-fold splitting
    @target_path: where the resulting folds should be stored
    @features: list of user features that will be kept in the dataset, defaults to None to keep all features
    @resampling_strategy: whether and how to perform resampling of the dataset
    @random_state: the random state for the experiments
    """
    state_file = os.path.join(target_path, "used_config.json")
    prev_state = None
    if os.path.exists(state_file):
        try:
            prev_state = json_load(state_file)
        except:
            pass

    current_state = {
        "random_state": random_state,
        "features": features,
    }

    states_match = (
        prev_state is not None
        and prev_state["random_state"] == current_state["random_state"]
        and prev_state["resampling_strategy"] == current_state["resampling_strategy"]
        and len(set(current_state["features"] or []) - set(prev_state["features"])) == 0
    )

    if True:
        interaction_matrix = sp.load_npz(os.path.join(data_dir, "interactions.npz"))
        df_user_info = pd.read_csv(os.path.join(data_dir, "user_info.csv"))

        features = features or list(df_user_info.columns)
        features_not_found = set(features) - set(df_user_info.columns)
        if len(features_not_found) > 0:
            raise AttributeError(
                f"Dataset does not contain the user features {features_not_found}"
            )
        current_state["features"] = features

        print(f"Preparing dataset for experiments on features {features}")
        perform_kfold_split(
            n_folds,
            interaction_matrix,
            df_user_info,
            storage_dir=target_path,
            features=features,
            random_state=random_state,
            drop_fold_items=drop_fold_items,
        )

        # store state so that we don't repeat processing the features
        json_dump(current_state, state_file)

def transform_dataframe_to_sparse(
    interactions: pd.DataFrame,
    item2token: pd.Series = pd.Series([]),
    user2token: pd.Series = pd.Series([]),
):
    """Generates a sparse matrix from a csv with user features

    Args:
        interactions (pd.DataFrame): _description_
        item2token (pd.Series, optional): _description_. Defaults to None.
        user2token (pd.Series, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if len(item2token) == 0:
        print(f"Creating item indexes")
        unique_items = interactions["itemID"].sort_values().unique()
        item2token = pd.Series(unique_items, name="itemID")
        item2token.index.name = "item_idx"
    token2item = pd.Series(data=item2token.index.values, index=item2token.values)
    if len(user2token) == 0:
        print(f"Creating user indexes")
        unique_users = interactions["userID"].unique()
        user2token = pd.Series(unique_users, name="userID")
        user2token.index.name = "user_idx"
    token2user = pd.Series(data=user2token.index.values, index=user2token.values)
    print(
        f"n_inter: {len(interactions)}: n_items: {len(item2token)} n_users: {len(user2token)}"
    )
    mapped_inter = interactions.copy()
    # assigning unique ids
    mapped_inter.loc[:, "userID"] = token2user.loc[mapped_inter["userID"]].values
    mapped_inter.loc[:, "itemID"] = token2item.loc[mapped_inter["itemID"]].values

    user_info = interactions.drop_duplicates(["userID"])

    uids_iids_array = mapped_inter[["userID", "itemID"]].values
    n_users, n_items = len(user2token), len(item2token)
    data = np.ones(uids_iids_array.shape[0], dtype=np.int8)

    uids, iids = uids_iids_array[:, 0], uids_iids_array[:, 1]
    interaction_matrix = sp.csr_matrix((data, (uids, iids)), (n_users, n_items))
    return interaction_matrix, user_info, item2token, user2token

### Preprocessing datasets
def read_ml1m(ROOT_DIR: str):
    print("Reading ml-1m dataset")
    print(os.path.isfile(os.path.join(ROOT_DIR, "ml-1m", "ratings.dat")))
    if not os.path.isfile(os.path.join(ROOT_DIR, "ml-1m", "ratings.dat")) or not os.path.isfile(os.path.join(ROOT_DIR, "ml-1m", "users.dat")):
        return None, None

    data_inter = pd.read_csv(
        os.path.join(ROOT_DIR, "ml-1m", "ratings.dat"),
        sep="::",
        names=["userID", "itemID", "rating", "timestamp"],
        engine="python",
    )
    data_user = pd.read_csv(
        os.path.join(ROOT_DIR, "ml-1m", "users.dat"),
        sep="::",
        names=["userID", "gender", "age", "occcupation", "zipcode"],
        engine="python",
    )
    return data_inter, data_user



def read_lfmdemobias(ROOT_DIR: str):
    # /lfm-demobias/sampled_100000_items_demo.txt
    # /lfm-demobias/sampled_100000_items_inter.txt
    # /lfm-demobias/sampled_100000_items_tracks.txt
    #check if the path exists
    if not os.path.isfile(os.path.join(ROOT_DIR, "lfm-demobias", "sampled_100000_items_demo.txt")) or not os.path.isfile(os.path.join(ROOT_DIR, "lfm-demobias", "sampled_100000_items_inter.txt")):
        return None, None
    data_inter = pd.read_csv(
        os.path.join(ROOT_DIR, "lfm-demobias", "sampled_100000_items_inter.txt"),
        sep="\t",
        header=None,
        names=["userID", "itemID", "pc"],
    )
    data_user = pd.read_csv(
        os.path.join(ROOT_DIR, "lfm-demobias", "sampled_100000_items_demo.txt"),
        delimiter="\t",
        names=["age", "gender"],
        usecols=[2, 3],
    )
    data_user["userID"] = data_user.index
    return data_inter, data_user


def read_ekstrabladet(ROOT_DIR: str):

    if not os.path.isfile(os.path.join(ROOT_DIR, "ekstrabladet", "interactions.csv")) or not os.path.isfile(os.path.join(ROOT_DIR, "ekstrabladet", "user_features.csv")):
        return None, None

    data_inter = pd.read_csv(
        os.path.join(ROOT_DIR, "ekstrabladet", "interactions.csv"),
    )
    data_user = pd.read_csv(
        os.path.join(ROOT_DIR, "ekstrabladet", "user_features.csv"),
    )
    map_gender = {0.0: "M", 1.0: "F"}
    data_user["gender"] = data_user["gender"].apply(lambda x: map_gender.get(x))

    return data_inter, data_user

def preprocess_data(
    data_dir: str,
    n_folds: int,
    target_path: str,
    random_state: int = 42,
    drop_fold_items=True,
):
    """
    Ensure that dataset is prepared for experiments by performing `n_folds` fold splitting for cross-validation and
    possible resampling.

    @data_dir: the path to the dataset
    @n_folds: the number of folds for k-fold splitting
    @target_path: where the resulting folds should be stored
    @features: list of user features that will be kept in the dataset, defaults to None to keep all features
    @resampling_strategy: whether and how to perform resampling of the dataset
    @random_state: the random state for the experiments
    """
    state_file = os.path.join(target_path, "used_config.json")
    prev_state = None
    if os.path.exists(state_file):
        try:
            prev_state = json_load(state_file)
        except:
            pass

    current_state = {"random_state": random_state}

    states_match = (
        prev_state is not None
        and prev_state["random_state"] == current_state["random_state"]
        and len(set(current_state["features"] or []) - set(prev_state["features"])) == 0
    )

    if True:
        interaction_matrix = sp.load_npz(os.path.join(data_dir, "interactions.csv"))
        df_user_info = pd.read_csv(os.path.join(data_dir, "user_info.csv"))

        features = features or list(df_user_info.columns)
        features_not_found = set(features) - set(df_user_info.columns)
        if len(features_not_found) > 0:
            raise AttributeError(
                f"Dataset does not contain the user features {features_not_found}"
            )

        perform_kfold_split(
            n_folds,
            interaction_matrix,
            df_user_info,
            storage_dir=target_path,
            random_state=random_state,
            drop_fold_items=drop_fold_items,
        )

        # store state so that we don't repeat processing the features
        json_dump(current_state, state_file)


def core_filtering(data: pd.DataFrame, min_k: int = 5):
    while True:
        item_user_counts = data.groupby(["itemID"])["userID"].nunique().reset_index()
        user_item_counts = data.groupby(["userID"])["itemID"].nunique().reset_index()

        valid_items = item_user_counts[item_user_counts["userID"] >= min_k][
            "itemID"
        ].values
        valid_users = user_item_counts[user_item_counts["itemID"] >= min_k][
            "userID"
        ].values

        result = data[
            data["itemID"].isin(valid_items) & data["userID"].isin(valid_users)
        ]

        item_user_counts = (
            result.groupby(["itemID"])["userID"].nunique().reset_index()["userID"]
        )
        user_item_counts = (
            result.groupby(["userID"])["itemID"].nunique().reset_index()["itemID"]
        )

        if (item_user_counts >= min_k).any() and (user_item_counts >= min_k).any():
            break
        else:
            data = result
            print("Iterate filtering")
            if len(data) < 1000:
                print("invalid")
                break
    return result

def preprocess_dataset(
    data_inter: pd.DataFrame,
    data_user: pd.DataFrame,
    ROOT_DIR: str,
    dataset_name: str,
    K_CORE: int = 5,
    item_features=["popularity", "item-stereo"],
):
    if data_inter is None or data_user is None:
        print(f"Dataset {dataset_name} not found, skipping preprocessing")
        return

    joined = data_inter.merge(data_user, on="userID").dropna()

    # Filtering dataset

    joined = core_filtering(joined, K_CORE)
    out_dir = os.path.join(ROOT_DIR, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Generate item features file
    # item_data = compute_item_features(joined, item_features)
    #
    print("Saving filtered dataset")

    interaction_matrix, user_info, item2token, user2token = (
        transform_dataframe_to_sparse(joined)
    )
    sp.save_npz(os.path.join(out_dir, "interactions.npz"), interaction_matrix)
    print(len(user_info), len(user2token))
    # mergin with ids
    item2token.to_csv(os.path.join(out_dir, "item_idx.csv"))
    user2token.to_csv(os.path.join(out_dir, "user_idx.csv"))
    user_idxs = user2token.reset_index()
    item_idxs = item2token.reset_index()
    user_info = user_info.merge(user_idxs)
    joined = joined.merge(user_idxs).merge(item_idxs)

    joined.to_csv(
        os.path.join(ROOT_DIR, dataset_name, f"{dataset_name}_filtered.csv"),
        index=False,
    )
    user_info.to_csv(os.path.join(out_dir, "user_info.csv"), index=False)
# %%

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "processed")
datasets = []
lfm_data, lfm_user_data = read_lfmdemobias(ROOT_DIR)

print("*****************LFM-DEMOBIAS*************************")
preprocess_dataset(
    lfm_data, lfm_user_data, ROOT_DIR=OUTPUT_DIR, dataset_name="lfm-demobias"
)
if lfm_data is not None and lfm_user_data is not None:
    datasets.append("lfm-demobias")
print("*****************EKSTRABLADET*************************")
eb_data, eb_user_data = read_ekstrabladet(ROOT_DIR)
preprocess_dataset(
    eb_data, eb_user_data, ROOT_DIR=OUTPUT_DIR, dataset_name="ekstrabladet"
)
if eb_data is not None and eb_user_data is not None:
    datasets.append("ekstrabladet")
print("*****************ML 1M*************************")
ml1m_data, ml1m_user_data = read_ml1m(ROOT_DIR)

preprocess_dataset(ml1m_data, ml1m_user_data, ROOT_DIR=OUTPUT_DIR, dataset_name="ml-1m")
if ml1m_data is not None and ml1m_user_data is not None:
    datasets.append("ml-1m")

# SPLITTING IN K-FOLDS


for dataset in datasets:
    data_dir = os.path.join(OUTPUT_DIR, dataset)
    n_folds = 5
    target_path = os.path.join(OUTPUT_DIR, dataset)
    ensure_make_data(
        data_dir,
        n_folds,
        target_path,
        drop_fold_items=False,
    )