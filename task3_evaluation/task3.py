"""Shared functions for 03_evaluation.ipynb (Task 3).

- ALS functions: copied from task2 (Task 2), without its top-level code
  (task2 runs the whole grid search as soon as it is executed or imported).
- Baselines and naive SVD: copied from the Task 1 part of the group notebook.
- Top-N ranking helpers: written for Task 3.

Changes to the copied code: the index mappings and the baseline averages are
passed as parameters instead of being read from globals, and fit_als also
works without validation data (R_val_coo=None).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import solve


def build_index_mappings(train):
    """0-based user/movie indices from the training set (sorted ids).
    The rows of U_best / V_best follow exactly this ordering."""
    unique_users = np.sort(train["user_id"].unique())
    unique_movies = np.sort(train["movie_id"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    movie_to_idx = {mid: j for j, mid in enumerate(unique_movies)}
    return unique_users, unique_movies, user_to_idx, movie_to_idx


def df_to_coo(df, user_to_idx, movie_to_idx):
    """Ratings dataframe -> sparse COO matrix of shape (n_users, n_movies)."""
    rows = df["user_id"].map(user_to_idx).to_numpy()
    cols = df["movie_id"].map(movie_to_idx).to_numpy()
    vals = df["rating"].to_numpy(dtype=np.float64)
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(user_to_idx), len(movie_to_idx)))


def solve_als_subproblem(fixed_factors, ratings_csr, lambda_reg):
    """
    Solves the ridge regression sub-problem for each row (user or movie)
    strictly over observed entries using Cholesky factorization (assume_a='pos').
    """
    n_targets = ratings_csr.shape[0]
    k = fixed_factors.shape[1]
    updated_factors = np.zeros((n_targets, k), dtype=np.float64)
    reg_eye = lambda_reg * np.eye(k, dtype=np.float64)

    indptr = ratings_csr.indptr
    indices = ratings_csr.indices
    data = ratings_csr.data

    for i in range(n_targets):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue

        obs_idx = indices[start:end]      # Indices of observed ratings in row i
        obs_ratings = data[start:end]     # Observed rating values r_{Omega_i}

        # Submatrix of fixed factors corresponding to observed entries
        F_sub = fixed_factors[obs_idx]    # Shape: (|Omega_i|, k)

        # Normal equation: (F_sub^T @ F_sub + lambda * I_k) x = F_sub^T @ r
        A = F_sub.T @ F_sub + reg_eye
        b = F_sub.T @ obs_ratings

        # Solve SPD system via Cholesky decomposition
        updated_factors[i] = solve(A, b, assume_a="pos")

    return updated_factors


def evaluate_metrics_sparse(R_coo, U, V):
    """Computes RMSE and MAE strictly over the observed entries in R_coo."""
    preds = np.sum(U[R_coo.row] * V[R_coo.col], axis=1)
    errors = preds - R_coo.data
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return rmse, mae


def fit_als(R_train_coo, R_val_coo=None, k=10, lambda_reg=5.0, max_iter=20, tol=1e-4, random_state=42, verbose=True):
    """
    Fits the low-rank factorization R ~ U V^T using Alternating Least Squares (ALS).
    Returns (U, V, history_df).
    """
    R_csr = R_train_coo.tocsr()
    R_T_csr = R_train_coo.tocsc().T.tocsr()
    n_users, n_movies = R_train_coo.shape

    rng = np.random.default_rng(random_state)
    # Initialize factors around the square root of the global mean for faster convergence
    global_mean = np.mean(R_train_coo.data)
    init_mean = np.sqrt(global_mean / k)
    U = init_mean + rng.normal(0.0, 0.1, size=(n_users, k))
    V = init_mean + rng.normal(0.0, 0.1, size=(n_movies, k))

    history = {"iter": [], "train_rmse": [], "val_rmse": [], "val_mae": []}

    for it in range(1, max_iter + 1):
        # Step 1: Fix V, solve for U
        U = solve_als_subproblem(V, R_csr, lambda_reg)

        # Step 2: Fix U, solve for V
        V = solve_als_subproblem(U, R_T_csr, lambda_reg)

        train_rmse, _ = evaluate_metrics_sparse(R_train_coo, U, V)
        history["iter"].append(it)
        history["train_rmse"].append(train_rmse)

        if R_val_coo is not None:
            val_rmse, val_mae = evaluate_metrics_sparse(R_val_coo, U, V)
            history["val_rmse"].append(val_rmse)
            history["val_mae"].append(val_mae)
            if verbose:
                print(f"Iteration {it:02d}/{max_iter:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")
        else:
            if verbose:
                print(f"Iteration {it:02d}/{max_iter:02d} | Train RMSE: {train_rmse:.4f}")

        if it > 1:
            rel_change = abs(history["train_rmse"][-2] - train_rmse) / history["train_rmse"][-2]
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iteration {it} (relative change {rel_change:.2e} < {tol}).")
                break

    # Without validation data the val columns stay empty, so drop them for a valid DataFrame
    if R_val_coo is None:
        history = {"iter": history["iter"], "train_rmse": history["train_rmse"]}
    return U, V, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Task 1 baselines (copied from the group notebook). Only change: the averages are
# passed as parameters instead of being read from globals.
# ---------------------------------------------------------------------------

def compute_baseline_averages(train):
    """Global, per-user and per-movie mean rating, computed on train only."""
    global_avg = train["rating"].mean()
    user_avg = train.groupby("user_id")["rating"].mean()
    movie_avg = train.groupby("movie_id")["rating"].mean()
    return global_avg, user_avg, movie_avg


def predict_baseline(data, global_avg, user_avg, movie_avg):
    predictions = data[["user_id", "movie_id", "rating"]].copy()

    # Everyone receives the same prediction
    predictions["global_avg"] = global_avg

    # Look up each user's training average
    predictions["user_avg"] = data["user_id"].map(user_avg)

    # Look up each movie's training average
    predictions["movie_avg"] = data["movie_id"].map(movie_avg)

    return predictions


def predict_svd(data, R, predicted_matrix):
    predictions = []

    for _, row in data.iterrows():
        # Find the user and movie positions
        user_position = R.index.get_loc(row["user_id"])
        movie_position = R.columns.get_loc(row["movie_id"])

        # Read the predicted rating
        prediction = predicted_matrix[user_position, movie_position]
        predictions.append(prediction)

    # Add predictions next to the actual ratings
    result = data[["user_id", "movie_id", "rating"]].copy()
    result["prediction"] = predictions

    return result


def find_best_k(validation, R, U, s, Vt, k_values):
    results = []

    for k in k_values:
        # Check that k is valid
        if k < 1 or k > len(s):
            raise ValueError(f"k must be between 1 and {len(s)}")

        # Reconstruct the matrix using k components
        U_small = U[:, :k]
        Sigma_small = np.diag(s[:k])
        Vt_small = Vt[:k, :]

        predicted_matrix = U_small @ Sigma_small @ Vt_small

        # Get predictions for the validation data
        predictions = predict_svd(validation, R, predicted_matrix)

        # Calculate both error metrics
        errors = predictions["prediction"] - predictions["rating"]

        rmse = np.sqrt((errors ** 2).mean())
        mae = errors.abs().mean()

        results.append({"k": k, "RMSE": rmse, "MAE": mae})

    # Sort by RMSE, then prefer the smaller k if errors are equal
    results = pd.DataFrame(results)
    results = results.sort_values(["RMSE", "k"]).reset_index(drop=True)

    best_k = int(results.loc[0, "k"])

    print("Best k:", best_k)
    print("Validation RMSE:", round(results.loc[0, "RMSE"], 4))
    print("Validation MAE:", round(results.loc[0, "MAE"], 4))

    return best_k, results


# ---------------------------------------------------------------------------
# Top-N ranking (Task 3)
# ---------------------------------------------------------------------------

def top_n_items(scores, exclude_mask, n):
    """Column indices of the n highest-scoring items per user (row), skipping excluded items."""
    scores = np.where(exclude_mask, -np.inf, scores)
    # Stable sort: ties are broken by item index, so the result is reproducible
    return np.argsort(-scores, axis=1, kind="stable")[:, :n]


def ranking_metrics(top_items, relevant_mask, users):
    """Precision@N, Recall@N and NDCG@N (binary relevance), averaged over the given users."""
    n = top_items.shape[1]
    hits = relevant_mask[users[:, None], top_items[users]]  # (len(users), n), True where the item is relevant
    n_relevant = relevant_mask[users].sum(axis=1)

    # DCG: a hit at position r (1-based) is worth 1 / log2(r + 1)
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = hits @ discounts
    # Ideal DCG: all relevant items at the top of the list (at most n of them fit)
    idcg = np.concatenate([[0.0], np.cumsum(discounts)])[np.minimum(n_relevant, n)]

    return {
        "Precision": float(np.mean(hits.sum(axis=1) / n)),
        "Recall": float(np.mean(hits.sum(axis=1) / n_relevant)),
        "NDCG": float(np.mean(dcg / idcg)),
    }
