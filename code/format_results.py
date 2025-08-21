import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE


def plot_pca_or_tsne_layerwise(
    X_pos,
    X_neg,
    hue,
    standardize=True,
    reshape=None,
    n_components=5,
    components=[0, 1],  # type: ignore
    mode="pca",
    plot_title=None,
):
    """
    Fixed PCA or T-SNE clustering with original plotting style to match first image

    Parameters:
        X_pos (np.ndarray): Positive samples, shape (n_samples, n_layers, hidden_dim)
        X_neg (np.ndarray): Negative samples, shape (n_samples, n_layers, hidden_dim)
        hue (np.ndarray | pd.Series): y values for coloring
        standardize (bool): If standardization is needed before PCA
        reshape (list): if data has size n_examples*n_layers, hidden_dim
        n_components (int): Number of PCA/TSNE components
        components (list): 2 components to plot
        mode (str): 'pca' or 'tsne'
        plot_title (str): figure suptitle
    """

    # Convert to numpy arrays with explicit checks
    if not isinstance(X_pos, np.ndarray):
        X_pos = np.array(X_pos, dtype=np.float32)
    else:
        X_pos = X_pos.astype(np.float32)

    if not isinstance(X_neg, np.ndarray):
        X_neg = np.array(X_neg, dtype=np.float32)
    else:
        X_neg = X_neg.astype(np.float32)

    if len(X_pos.shape) == 2:
        if reshape is None:
            reshape = [
                int(i)
                for i in input("Get reshape params (len data, n_layers)").split(",")
            ]
        X_pos = X_pos.reshape(reshape[0], reshape[1], -1)
        X_neg = X_neg.reshape(reshape[0], reshape[1], -1)

    n_layers = X_pos.shape[1]

    # Fixed subplot calculation logic with explicit checks
    if (n_layers - 1) % 6 == 0:
        n_rows = (n_layers - 1) // 6
    else:
        n_rows = (n_layers - 1) // 6 + 1

    #  Ensure minimum 1 row
    if n_rows == 0:
        n_rows = 1

    fig, axes = plt.subplots(n_rows, 6, figsize=(24, 13))

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for layer_idx in range(1, n_layers):
        ax = axes[layer_idx - 1]

        X_pos_layer = X_pos[:, layer_idx, :]
        X_neg_layer = X_neg[:, layer_idx, :]

        # Combining hidden states of a model to plot a graph
        states_data = X_pos_layer - X_neg_layer

        #  Enhanced standardization with explicit variance checking
        if standardize:
            data_mean = np.mean(states_data, axis=0)
            data_std = np.std(states_data, axis=0)
            # Explicit check for zero variance features
            nonzero_std_mask = data_std > 1e-10
            data_std_safe = np.where(nonzero_std_mask, data_std, 1.0)
            states_data = (states_data - data_mean) / data_std_safe

        #  Enhanced numerical stability checks for PCA
        if mode == "pca":
            feature_vars = np.var(states_data, axis=0)
            non_constant_features = feature_vars > 1e-10

            if np.sum(non_constant_features) > 0:
                states_data_filtered = states_data[:, non_constant_features]
                actual_n_components = min(
                    n_components,
                    states_data_filtered.shape[1],
                    states_data_filtered.shape[0],
                )

                # Explicit check for valid n_components
                if actual_n_components > 0:
                    projector = PCA(n_components=actual_n_components)
                    X_proj_filtered = projector.fit_transform(states_data_filtered)

                    # Pad with zeros if needed
                    if X_proj_filtered.shape[1] < n_components:
                        X_proj = np.zeros((X_proj_filtered.shape[0], n_components))
                        X_proj[:, : X_proj_filtered.shape[1]] = X_proj_filtered
                    else:
                        X_proj = X_proj_filtered
                else:
                    X_proj = np.zeros((states_data.shape[0], n_components))
            else:
                X_proj = np.zeros((states_data.shape[0], n_components))

        elif mode == "tsne":
            #  Add explicit check for valid data
            if (
                states_data.shape[0] > 1
                and np.sum(np.var(states_data, axis=0) > 1e-10) > 1
            ):
                projector = TSNE(
                    n_components=min(n_components, 2), metric="cosine", random_state=42
                )
                X_proj_tsne = projector.fit_transform(states_data)
                # Pad if needed
                if X_proj_tsne.shape[1] < n_components:
                    X_proj = np.zeros((X_proj_tsne.shape[0], n_components))
                    X_proj[:, : X_proj_tsne.shape[1]] = X_proj_tsne
                else:
                    X_proj = X_proj_tsne
            else:
                X_proj = np.zeros((states_data.shape[0], n_components))

        elif mode == "pca-sparse":
            feature_vars = np.var(states_data, axis=0)
            non_constant_features = feature_vars > 1e-10

            if np.sum(non_constant_features) > 0:
                states_data_filtered = states_data[:, non_constant_features]
                actual_n_components = min(
                    n_components,
                    states_data_filtered.shape[1],
                    states_data_filtered.shape[0],
                )

                if actual_n_components > 0:
                    projector = SparsePCA(
                        n_components=actual_n_components, alpha=0, random_state=42
                    )
                    X_proj_filtered = projector.fit_transform(states_data_filtered)

                    if X_proj_filtered.shape[1] < n_components:
                        X_proj = np.zeros((X_proj_filtered.shape[0], n_components))
                        X_proj[:, : X_proj_filtered.shape[1]] = X_proj_filtered
                    else:
                        X_proj = X_proj_filtered
                else:
                    X_proj = np.zeros((states_data.shape[0], n_components))
            else:
                X_proj = np.zeros((states_data.shape[0], n_components))

        # CHANGED BACK TO ORIGINAL: Bounds checking for components but using original plotting style
        component_0 = min(components[0], X_proj.shape[1] - 1)
        component_1 = min(components[1], X_proj.shape[1] - 1)

        # CHANGED BACK TO ORIGINAL: Original plotting style - each subplot auto-scaled independently
        if X_proj.shape[1] > max(component_0, component_1):
            # CHANGED BACK TO ORIGINAL: Use original seaborn scatterplot style
            sns.scatterplot(
                data=pd.DataFrame(X_proj),
                x=X_proj[:, component_0],
                y=X_proj[:, component_1],
                hue=hue,
                ax=ax,
            )
        else:
            # Handle case where projection has insufficient components
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\nfor Layer {layer_idx}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # CHANGED BACK TO ORIGINAL: Original styling to match first image
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.legend().set_visible(False)
        ax.set_xticks([])  # CHANGED BACK TO ORIGINAL: Remove tick marks
        ax.set_yticks([])  # CHANGED BACK TO ORIGINAL: Remove tick marks
        ax.grid(True)  # CHANGED BACK TO ORIGINAL: Simple grid

    # CHANGED BACK TO ORIGINAL: Turn off unused axes
    for idx in range(n_layers - 1, len(axes)):
        axes[idx].axis("off")

    # CHANGED BACK TO ORIGINAL: Original legend style and placement
    if n_layers > 1:
        # Get handles and labels from the last plotted axis
        handles, labels = axes[n_layers - 2].get_legend_handles_labels()

        # CHANGED BACK TO ORIGINAL: Use original legend logic
        if hasattr(hue, "name") and hue.name is not None:
            title = hue.name
        else:
            title = None

        fig.legend(handles, labels, loc="upper right", fontsize=12, title=title)

    # CHANGED BACK TO ORIGINAL: Original title style
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)

    # CHANGED BACK TO ORIGINAL: Close all figures like original
    plt.close("all")

    return fig


def get_results_table(ccs_results):
    acc_list = []
    slh_list = []
    agreement_list = []
    agreement_abs_list = []
    ci_list = []

    for layer in ccs_results.keys():
        acc_list.append(ccs_results[layer]["accuracy"])
        slh_list.append(ccs_results[layer]["silhouette"])
        agreement_list.append(np.mean(ccs_results[layer]["agreement"]))
        agreement_abs_list.append(np.median(np.abs(ccs_results[layer]["agreement"])))
        ci_list.append(np.mean(ccs_results[layer]["contradiction idx"]))

    data = pd.DataFrame(
        index=ccs_results.keys(),
        data=np.array(
            [acc_list, agreement_list, agreement_abs_list, ci_list, im_dist_list]
        ).T,
        columns=[
            "accuracy",
            "polar_consistency_↓",
            "abs_agreement_score",
            "contradiction_idx_↓",
        ],
    )
    data["layer_number"] = range(1, len(data) + 1)
    data["relative_postion"] = round((data["layer_number"] - 1) / len(data) * 100)

    return data
