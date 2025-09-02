import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr

def plot_umap_with_insets_multiple(
    adata: sc.AnnData,
    basis1: str = 'X_harmony',
    comp1_list: list = [0, 0, 0],
    basis2: str = 'X_ahba_harmony',
    comp2_list: list = [4, 5, 6],
    color: str = 'Cell_Lineage',
    color_map: str = 'Spectral',
    subset_cell_type: str = 'L2-3',
    subset_coords_x: list = [0.6, 0.6, 0.6],
    subset_coords_y: list = [0.6, 0.6, 0.6],
    inset_color: str = 'Cell_Lineage',
    figure_size: tuple = (5, 6), # Base size per panel
    main_title: str = None, # New parameter for overall title
    xlabels: list[str] = None, # New parameter for custom x-axis labels
    ylabels: list[str] = None # New parameter for custom y-axis labels
):
    """
    Plot multiple UMAP panels, each with an inset showing a subset of the data.
    Each panel uses a pair of components from comp1_list and comp2_list.
    Adds Pearson correlation (R) to the top-left corner of both main and inset plots.

    Parameters:
    - adata: AnnData object containing the data.
    - basis1: The first embedding basis (e.g., 'X_pca', 'X_harmony').
    - comp1_list: A list of component indices from basis1 to use for the X-axis
                  for each main panel. Must be same length as comp2_list.
    - basis2: The second embedding basis (e.g., 'X_ahba', 'X_ahba_harmony').
    - comp2_list: A list of component indices from basis2 to use for the Y-axis
                  for each main panel. Must be same length as comp1_list.
    - color: The column in adata.obs to use for coloring the main plot points
             (e.g., 'Cell_Lineage').
    - palette: The color palette to use for the main plot (e.g., 'Set1').
    - color_map: The color map to use for the main plot (e.g., 'Spectral').
    - subset_cell_type: The value in adata.obs['Cell_Type'] to subset for the inset.
    - inset_color: The column in adata.obs to use for coloring the inset plot points
                   (e.g., 'Age_log2').
    - figure_size: A tuple (width, height) specifying the base size for each panel.
                   The total figure width will be figure_size[0] * num_panels.
    - main_title: Optional string for a single title across all plots.
    - xlabels: Optional list of strings for custom x-axis labels for each panel.
               If provided, length must match num_plots.
    - ylabels: Optional list of strings for custom y-axis labels for each panel.
               If provided, length must match num_plots.
    """

    # --- Input Validation ---
    if len(comp1_list) != len(comp2_list):
        raise ValueError("comp1_list and comp2_list must have the same number of elements.")
    num_plots = len(comp1_list) # Or len(comp2_list), they are guaranteed to be equal
    if xlabels is not None and len(xlabels) != num_plots:
        raise ValueError("If 'xlabels' is provided, its length must match the number of plots (length of comp1_list).")
    if ylabels is not None and len(ylabels) != num_plots:
        raise ValueError("If 'ylabels' is provided, its length must match the number of plots (length of comp2_list).")
    # --- End Input Validation ---

    adata = adata.copy()
    adata.obs['Cell_Lineage'] = adata.obs['Cell_Lineage'].cat.remove_unused_categories()

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Create the subset AnnData object once
    if subset_cell_type not in adata.obs['Cell_Type'].unique():
        print(f"Warning: Cell_Type value '{subset_cell_type}' not found in adata.obs['Cell_Type']. Inset will be empty or not plotted.")
        subset_adata = sc.AnnData(np.array([])) # Create an empty AnnData for safe handling
    else:
        subset_adata = adata[adata.obs['Cell_Type'] == subset_cell_type].copy()

    # Calculate total figure width, allowing space for a single grouped legend
    total_figure_width = figure_size[0] * num_plots + 3 # Extra 3 inches for legend
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(total_figure_width, figure_size[1]))

    # Ensure 'axes' is an array even if num_plots is 1, for consistent iteration
    if num_plots == 1:
        axes = [axes]

    # Store handles and labels for the grouped legend
    legend_handles = []
    legend_labels = []

    # Set a common font size
    default_fontsize = 20

    # Add overall title if provided
    if main_title:
        fig.suptitle(main_title, fontsize=default_fontsize + 2, y=1) # Slightly larger title

    # Loop through each pair of components to create a panel
    for i, (current_comp1, current_comp2) in enumerate(zip(comp1_list, comp2_list)):
        ax = axes[i] # Get the current axis for this panel

        # Prepare data for plotting and correlation calculation
        x_main = adata.obsm[basis1][:, current_comp1]
        y_main = adata.obsm[basis2][:, current_comp2]

        # Create the custom embedding for the current panel
        adata.obsm['X_plot'] = np.stack([x_main, y_main], axis=1)

        # Plot the main embedding on the current axis
        plot_kwargs = dict(
            basis='plot', color=color, palette=palette, color_map=color_map,
            title='',
            show=False, ax=ax, legend_loc=None # Suppress legend on subplot axes
        )
        sc.pl.embedding(adata, **plot_kwargs)

        # --- Add Pearson Correlation to Main Plot ---
        if len(x_main) > 1 and np.std(x_main) > 0 and np.std(y_main) > 0:
            r_main, _ = pearsonr(x_main, y_main)
            ax.text(0.05, 0.95, f'R = {r_main:.2f}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=default_fontsize, weight='bold', color='blue')
        else:
            ax.text(0.05, 0.95, 'R = N/A', transform=ax.transAxes,
                    ha='left', va='top', fontsize=default_fontsize, weight='bold', color='gray')
        # -------------------------------------------

        # Manually set the x and y axis labels for the main plot
        if xlabels is not None:
            ax.set_xlabel(xlabels[i], fontsize=default_fontsize)
        else:
            ax.set_xlabel(f'{basis1.replace("X_", "").capitalize()} Component {current_comp1 + 1}', fontsize=default_fontsize)

        if ylabels is not None:
            ax.set_ylabel(ylabels[i], fontsize=default_fontsize)
        else:
            ax.set_ylabel(f'{basis2.replace("X_", "").capitalize()} Component {current_comp2 + 1}', fontsize=default_fontsize)

        # Remove subtitles on panels (already handled in previous iteration, ensuring no ax.set_title here)

        # Add inset if subset data exists
        if subset_adata.n_obs > 0:
            inset_ax = ax.inset_axes([subset_coords_x[i], subset_coords_y[i], 0.35, 0.35])

            # Prepare data for inset plotting and correlation calculation
            x_inset = subset_adata.obsm[basis1][:, current_comp1]
            y_inset = subset_adata.obsm[basis2][:, current_comp2]

            # The inset also needs its X_plot re-stacked based on the current components
            subset_adata.obsm['X_plot'] = np.stack([x_inset, y_inset], axis=1)

            sc.pl.embedding(subset_adata, basis='plot', size=8,
                            color=inset_color, color_map=color_map,
                            ax=inset_ax, show=False, legend_loc=None,
                            colorbar_loc=None) # Suppress colorbar for inset

            # --- Add Pearson Correlation to Inset Plot ---
            if len(x_inset) > 1 and np.std(x_inset) > 0 and np.std(y_inset) > 0:
                r_inset, _ = pearsonr(x_inset, y_inset)
                inset_ax.text(0.05, 0.95, f'R = {r_inset:.2f}', transform=inset_ax.transAxes,
                              ha='left', va='top', fontsize=default_fontsize, color='blue', weight='bold') # Slightly smaller font for inset R
            else:
                inset_ax.text(0.05, 0.95, 'R = N/A', transform=inset_ax.transAxes,
                              ha='left', va='top', fontsize=default_fontsize, color='gray')
            # ---------------------------------------------

            inset_ax.set_title(f'{subset_cell_type} cells only', fontsize=default_fontsize, pad=5) # Increased fontsize, added pad
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlabel('')
            inset_ax.set_ylabel('')
            inset_ax.tick_params(labelsize=default_fontsize) # Ensure ticks are also affected if visible
            inset_ax.patch.set_edgecolor('black')
            inset_ax.patch.set_linewidth(1)
        else:
            ax.text(0.5, 0.75, f"No '{subset_cell_type}' cells for inset",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=default_fontsize, color='gray')

        # Collect legend handles and labels from a temporary plot for the grouped legend
        if i == 0:
            # try:
            #     n_colors = adata.obs[color].nunique()
            #     cmap = plt.get_cmap(palette)
            #     if hasattr(cmap, 'colors'):
            #         discrete_cmap = ListedColormap(cmap.colors[:n_colors])
            #     else: # For continuous colormaps, sample evenly
            #         discrete_cmap = cmap

            #     for j, category in enumerate(adata.obs[color].unique()):
            #         norm_idx = j / (n_colors - 1) if n_colors > 1 else 0
            #         color_val = discrete_cmap(norm_idx)
            #         legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
            #                                          markerfacecolor=color_val, markersize=10))
            #         legend_labels.append(category)
            # except Exception as e:
                # print(f"Warning: Could not robustly extract colors for grouped legend from palette '{palette}'. Error: {e}")
                # print("Legend colors might be inconsistent. Fallback to default legend handles.")
                # Fallback to general handles/labels from a temporary plot for safety
            temp_plot_ax = fig.add_subplot(111, frameon=False, visible=False) # Create a temp axis to get legend
            sc.pl.embedding(adata, basis='plot', color=color, palette=palette, show=False, ax=temp_plot_ax)
            legend_handles, legend_labels = temp_plot_ax.get_legend_handles_labels()
            fig.delaxes(temp_plot_ax) # Delete the temporary axis


    # --- Create a single, grouped legend outside the subplot loop ---
    # We set right to 0.9 matching the rect below, so subplots take 90%
    fig.subplots_adjust(right=0.9) # Right edge of subplots at 90% of figure width
    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='center right', title=color, bbox_to_anchor=(1.0, 0.3),
               fontsize=default_fontsize, title_fontsize=default_fontsize) # Consistent legend font sizes

    # --- Final plot layout adjustment ---
    # rect=[left, bottom, right, top] for the area tight_layout should consider
    plt.tight_layout(rect=[0, 0, 0.83, 1]) # Arrange subplots in the left 90% of the figure
    plt.show()




def plot_umap_with_inset(
    adata: sc.AnnData, 
    basis1: str = 'X_pca',
    comp1: int = 0,
    basis2: str = 'X_ahba',
    comp2: int = 4,
    color: str = 'Cell_Lineage', 
    palette: str = 'Set1', 
    color_map: str = 'Spectral'
):
    """
    Plot a UMAP with an inset showing a subset of the data.
    Parameters:
    - adata: AnnData object containing the data.
    - color: The column in adata.obs to use for coloring the points (e.g., 'Cell_Lineage').
    - palette: The color palette to use for the plot (e.g., 'Set1').
    - color_map: The color map to use for the plot (e.g., 'Spectral').
    """
    adata.obsm['X_plot'] = np.stack([
        adata.obsm[basis1][:, comp1], 
        adata.obsm[basis2][:, comp2]
    ], axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    # ax.set_aspect('equal', adjustable='box')
    sc.pl.embedding(adata, basis='plot', color=color, 
                    palette=palette, color_map=color_map, 
                    show=False, ax=ax)

    subset_adata = adata[adata.obs['Cell_Type'] == 'L2-3'].copy()
    inset_ax = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    sc.pl.embedding(subset_adata, basis='plot',
                    color='Age_log2', color_map='Spectral',
                    # color='Cell_Lineage', palette='Set1', color_map='Spectral',
                    ax=inset_ax, show=False, legend_loc=None)
    inset_ax.set_title('L2-3 cells only', fontsize=10) # Smaller font for subplot title
    inset_ax.set_xticks([]) # Remove x-axis ticks
    inset_ax.set_yticks([]) # Remove y-axis ticks
    inset_ax.set_xlabel('') # Remove x-axis label
    inset_ax.set_ylabel('') # Remove y-axis label
    inset_ax.patch.set_edgecolor('black') # Add a border for clarity
    inset_ax.patch.set_linewidth(1)

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
