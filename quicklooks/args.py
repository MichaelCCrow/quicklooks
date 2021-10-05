import argparse
import json
import numpy as np
from quicklooks.actions import timeseries, mtimeseries, skewt, geodisplay, contour, xsection, histogram, wind_rose


def getparser():
    parser = argparse.ArgumentParser(add_help=False)
    qlgroup = parser.add_argument_group('quicklooks arguments')
    parser.add_argument('-fds', '--fields', nargs='+', type=str, default=None,
                        help='Name of the fields to use to plot')
    parser.add_argument('-wfs', '--wind_fields', nargs='+', type=str, default=None,
                        help='Wind field names used to plot')
    parser.add_argument('-sfs', '--station_fields', nargs='+', type=str, default=None,
                        help='Station field names to plot sites')
    parser.add_argument('-lat', '--latitude', type=str, default='lat',
                        help='Name of latitude variable in file')
    parser.add_argument('-lon', '--longitude', type=str, default='lon',
                        help='Name of longitude variable in file')
    parser.add_argument('-xf', '--x_field', type=str, default=None,
                        help='Name of variable to plot on x axis')
    parser.add_argument('-yf', '--y_field', type=str, default=None,
                        help='Name of variable to plot on y axis')
    parser.add_argument('-x', type=np.array,
                        help='x coordinates or grid for z')
    parser.add_argument('-y', type=np.array,
                        help='y coordinates or grid for z')
    parser.add_argument('-z', type=np.array,
                        help='Values over which to contour')
    parser.add_argument('-u', '--u_wind', type=str, default='u_wind',
                        help='File variable name for u_wind wind component')
    parser.add_argument('-v', '--v_wind', type=str, default='v_wind',
                        help='File variable name for v_wind wind compenent')
    parser.add_argument('-pf', '--p_field', type=str, default=None,
                        help='File variable name for pressure')
    parser.add_argument('-tf', '--t_field', type=str, default='tdry',
                        help='File variable name for temperature')
    parser.add_argument('-tdf', '--td_field', type=str, default='dp',
                        help='File variable name for dewpoint temperature')
    parser.add_argument('-sf', '--spd_field', type=str, default='wspd',
                        help='File variable name for wind speed')
    parser.add_argument('-df', '--dir_field', type=str, default='deg',
                        help='File variable name for wind direction')
    parser.add_argument('-al', '--alt_label', type=str, default=None,
                        help='Altitude axis label')
    parser.add_argument('-af', '--alt_field', type=str, default='alt',
                        help='File variable name for altitude')
    parser.add_argument('-ds', '--dsname', type=str, default='act_datastream',
                        help='Name of datastream to plot')
    parser.add_argument('-vn', '--varname', type=str,
                        help='Name of the variable to plot')
    parser.add_argument('-cbl', '--cb_label', type=str, default=None,
                        help='Colorbar label to use')
    parser.add_argument('-st', '--set_title', type=str, default=None,
                        help='Title for the plot')
    parser.add_argument('-pb', '--plot_buffer', type=float, default=0.08,
                        help=('Buffer to add around data on plot in lat'
                              + 'and lon dimension'))
    parser.add_argument('-sm', '--stamen', type=str, default='terrain-background',
                        help='Dataset to use for background image')
    parser.add_argument('-tl', '--tile', type=int, default=8,
                        help='Tile zoom to use with background image')
    parser.add_argument('-cfs', '--cfeatures', nargs='+', type=str, default=None,
                        help='Cartopy feature to add to plot')
    parser.add_argument('-txt', '--text', type=json.loads, default=None,
                        help=('Dictionary of {text:[lon,lat]} to add to plot.'
                              + 'Can have more than one set of text to add.'))
    parser.add_argument('-cm', '--cmap', default='rainbow',
                        help='colormap to use')
    parser.add_argument('-nd', '--num_dir', type=int, default=20,
                        help='Number of directions to splot the wind rose into.')
    parser.add_argument('-sb', '--spd_bins', nargs='+', type=float, default=None,
                        help='Bin boundaries to sort the wind speeds into')
    parser.add_argument('-ti', '--tick_interval', type=int, default=3,
                        help=('Interval (in percentage) for the ticks'
                              + 'on the radial axis'))
    parser.add_argument('-bx', '--num_barb_x', type=int, default=20,
                        help='Number of wind barbs to plot in the x axis')
    parser.add_argument('-by', '--num_barb_y', type=int, default=20,
                        help='Number of wind barbs to plot in the y axis')
    parser.add_argument('-tp', '--num_time_periods', type=int, default=20,
                        help='Set how many time periods')
    parser.add_argument('-bn', '--bins', type=int, default=None,
                        help='histogram bin boundaries to use')
    parser.add_argument('-xb', '--x_bins', type=int, default=None,
                        help='Histogram bin boundaries to use for x axis variable')
    parser.add_argument('-yb', '--y_bins', type=int, default=None,
                        help='Histogram bin boundaries to use for y axis variable')
    parser.add_argument('-t', '--time', type=str, default=None,
                        help='Time period to be plotted')
    parser.add_argument('-sby', '--sortby_field', type=str, default=None,
                        help='Sort histograms by a given field parameter')
    parser.add_argument('-sbb', '--sortby_bins', type=int, default=None,
                        help='Bins to sort the histograms by')
    parser.add_argument('-nyl', '--num_y_levels', type=int, default=20,
                        help='Number of levels in the y axis to use')
    parser.add_argument('-sk', '--sel_kwargs', type=json.loads, default=None,
                        help=('The keyword arguments to pass into'
                              + ':py:func:`xarray.DataArray.sel`'))
    parser.add_argument('-ik', '--isel_kwargs', type=json.loads, default=None,
                        help=('The keyword arguments to pass into'
                              + ':py:func:`xarray.DataArray.sel`'))
    parser.add_argument('-fn', '--function', type=str, default='cubic',
                        help=('Defaults to cubic function for interpolation.'
                              + 'See scipy.interpolate.Rbf for additional options'))
    parser.add_argument('-gb', '--grid_buffer', type=float, default=0.1,
                        help='Buffer to apply to grid')
    parser.add_argument('-gd', '--grid_delta', nargs='+',
                        type=float, default=(0.01, 0.01),
                        help='X and Y deltas for creating grid')
    parser.add_argument('-fg', '--figsize', nargs='+', type=float,
                        default=None,
                        help='Width and height in inches of figure')
    parser.add_argument('-tc', '--text_color', type=str, default='white',
                        help='Color of text')
    parser.add_argument('-kwargs', type=json.loads,
                        help='keyword arguments to use in plotting function')
    parser.add_argument('-gl', '--gridlines', default=False, action='store_true',
                        help='Use latitude and lingitude gridlines.')
    parser.add_argument('-cl', '--coastlines', default=False, action='store_true',
                        help='Plot coastlines on geographical map')
    parser.add_argument('-bg', '--background', default=False, action='store_true',
                        help='Plot a stock image background')
    parser.add_argument('-n', '--add_nan', default=False, action='store_true',
                        help='Fill in data gaps with NaNs')
    parser.add_argument('-dn', '--day_night', default=False, action='store_true',
                        help='Fill in color coded background according to time of day')
    parser.add_argument('-yr', '--set_yrange', default=None, nargs=2,
                        help=("Set the yrange for the specific plot"))
    parser.add_argument('-iya', '--invert_y_axis', default=False, action='store_true',
                        help='Invert y axis')
    parser.add_argument('-d', '--density', default=False, action='store_true',
                        help='Plot a p.d.f. instead of a frequency histogram')
    parser.add_argument('-pq', '--plot_quartiles', default=False, action='store_true',
                        help='')
    parser.add_argument('-m', '--mesh', default=False, action='store_true',
                        help=('Set to True to interpolate u and v to'
                              + 'grid and create wind barbs'))
    parser.add_argument('-uv', '--from_u_and_v', default=False, action='store_true',
                        help='Create SkewTPLot with u and v wind')
    parser.add_argument('-sd', '--from_spd_and_dir', default=False, action='store_true',
                        help='Create SkewTPlot with wind speed and direction')
    parser.add_argument('-px', '--plot_xsection', default=False, action='store_true',
                        help='plots a cross section whose x and y coordinates')
    parser.add_argument('-pxm', '--xsection_map', default=False, action='store_true',
                        help='plots a cross section of 2D data on a geographical map')
    qlgroup.add_argument('-p', '--plot', default=False, action='store_true',
                         help='Makes a time series plot')
    parser.add_argument('-bsd', '--barbs_spd_dir', default=False, action='store_true',
                        help=('Makes time series plot of wind barbs'
                              + 'using wind speed and dir.'))
    parser.add_argument('-buv', '--barbs_u_v', default=False, action='store_true',
                        help=('Makes time series plot of wind barbs'
                              + 'using u and v wind components.'))
    parser.add_argument('-pxs', '--xsection_from_1d', default=False,
                        action='store_true',
                        help='Will plot a time-height cross section from 1D dataset')
    parser.add_argument('-ths', '--time_height_scatter',
                        default=False, action='store_true',
                        help='Create a scatter time series plot')
    parser.add_argument('-sbg', '--stacked_bar_graph',
                        default=False, action='store_true',
                        help='Create stacked bar graph histogram')
    parser.add_argument('-psd', '--size_dist', default=False, action='store_true',
                        help='Plots a stairstep plot of size distribution')
    parser.add_argument('-sg', '--stairstep', default=False, action='store_true',
                        help='Plots stairstep plot of a histogram')
    parser.add_argument('-hm', '--heatmap', default=False, action='store_true',
                        help='Plot a heatmap histogram from 2 variables')
    parser.add_argument('-cc', '--create_contour', default=False, action='store_true',
                        help='Extracts, grids, and creates a contour plot')
    parser.add_argument('-cf', '--contourf', default=False, action='store_true',
                        help=('Base function for filled contours if user'
                              + 'already has data gridded'))
    parser.add_argument('-ct', '--plot_contour', default=False, action='store_true',
                        help=('Base function for contours if user'
                              + 'already has data gridded'))
    parser.add_argument('-vsd', '--vectors_spd_dir', default=False, action='store_true',
                        help='Extracts, grids, and creates a contour plot.')
    parser.add_argument('-b', '--barbs', default=False, action='store_true',
                        help='Base function for wind barbs.')
    parser.add_argument('-ps', '--plot_station', default=False, action='store_true',
                        help='Extracts, grids, and creates a contour plot')
    parser.add_argument('-gp', '--geodisplay', dest='action',
                        action='store_const', const=geodisplay)
    parser.add_argument('-skt', '--skewt', dest='action',
                        action='store_const', const=skewt)
    parser.add_argument('-xs', '--xsection', dest='action',
                        action='store_const', const=xsection)
    parser.add_argument('-wr', '--wind_rose', dest='action',
                        action='store_const', const=wind_rose)
    qlgroup.add_argument('-ts', '--timeseries', dest='action',
                         action='store_const', const=timeseries)
    parser.add_argument('-mts', '--mtimeseries', dest='action',
                        action='store_const', const=mtimeseries)
    parser.add_argument('-c', '--contour', dest='action',
                        action='store_const', const=contour)
    parser.add_argument('-hs', '--histogram', dest='action',
                        action='store_const', const=histogram)
    return parser