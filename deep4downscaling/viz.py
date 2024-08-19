import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import math

def simple_map_plot(data: xr.Dataset, var_to_plot: str, output_path: str,
                    colorbar: str='coolwarm', vlimits: tuple=(None, None),
                    num_levels: int=20,
                    coord_names: dict={'lat': 'lat',
                                       'lon': 'lon'}) -> None:

    """
    This function generates a simple plot of a specific variable from a xr.DataArray
    or xr.Dataset.

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to plot. It is important this it does not have a temporal
        dimensions. otherwise this function will show an error.

    var_to_plot : str
        Variable to plot from the xr.Dataset. If data is a xr.DataArray it will
        ignore this parameter.

    output_path : str
        Path inidicating where to save the resulting image (pdf)

    colorbar : str, optional
        Colorbar to use in the plot (inherited from matplotlib)

    vlimits : tuple, optional
        Limits of the colorbar of the plot. If not indicated this will be computed
        by default.

    num_levels : int, optional
        The amount of levels to use in the colorbar. By default is 20.

    coord_names : dict, optional
        Dictionary with mappings of the name of the spatial dimensions.
        By default lat and lon.

    Returns
    -------
    None
    """                   

    if isinstance(data, xr.Dataset):
        data = data[var_to_plot]

    continuous_cmap = plt.get_cmap(colorbar)
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))    

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if None in vlimits:
        cs = ax.pcolormesh(data[coord_names['lon']], data[coord_names['lat']],
                        data, transform=ccrs.PlateCarree(),
                        cmap=discrete_cmap)
    else:
        cs = ax.pcolormesh(data[coord_names['lon']], data[coord_names['lat']],
                        data, transform=ccrs.PlateCarree(),
                        cmap=discrete_cmap,
                        vmin=vlimits[0], vmax=vlimits[1])

    ax.coastlines(resolution='10m')
    plt.colorbar(cs, ax=ax, orientation='horizontal')

    plt.title(var_to_plot)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def multiple_map_plot(data: xr.Dataset, output_path: str,
                      colorbar: str='coolwarm', vlimits: tuple=(None, None),
                      num_levels: int=20,
                      coord_names: dict={'lat': 'lat',
                                         'lon': 'lon'}) -> None:           

    """
    This function generates a plot of all variables from a xr.Dataset.

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to plot. It is important this it does not have a temporal
        dimensions. otherwise this function will show an error.

    output_path : str
        Path inidicating where to save the resulting image (pdf)

    colorbar : str, optional
        Colorbar to use in the plot (inherited from matplotlib)

    vlimits : tuple, optional
        Limits of the colorbar of the plot. If not indicated this will be computed
        by default.

    num_levels : int, optional
        The amount of levels to use in the colorbar. By default is 20.

    coord_names : dict, optional
        Dictionary with mappings of the name of the spatial dimensions.
        By default lat and lon.

    Returns
    -------
    None
    """       

    continuous_cmap = plt.get_cmap(colorbar)
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))    

    num_variables = len(data.keys())

    if num_variables % 2 == 1:
        num_rows, num_cols = math.ceil(num_variables/2), math.floor(num_variables/2)
    else:
        num_rows, num_cols = num_variables // 2, 2

    fig = plt.figure(figsize=(20, 20))

    plot_counter = 1
    for idx, var_to_plot in enumerate(data.keys()):
        
        ax = fig.add_subplot(num_rows, num_cols, plot_counter,
                             projection=ccrs.PlateCarree())

        data_to_plot = data[var_to_plot]

        if None in vlimits:
            cs = ax.pcolormesh(data_to_plot[coord_names['lon']], data_to_plot[coord_names['lat']],
                               data_to_plot, transform=ccrs.PlateCarree(),
                               cmap=discrete_cmap)
        else:
            cs = ax.pcolormesh(data_to_plot[coord_names['lon']], data_to_plot[coord_names['lat']],
                               data_to_plot, transform=ccrs.PlateCarree(),
                               cmap=discrete_cmap,
                               vmin=vlimits[0], vmax=vlimits[1])

        ax.set_title(var_to_plot)

        ax.coastlines(resolution='10m')
        
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        plt.colorbar(cs, cax=ax_cb, orientation='vertical')

        plot_counter = plot_counter+1
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()