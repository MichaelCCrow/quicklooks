import os
import math
import glob
import act
import pyart
from matplotlib import pyplot as plt


def timeseries(args):
    # ds = act.io.armfiles.read_netcdf(args.file_path)
    ds = args.dataset
    if args.plot:
        display = act.plotting.TimeSeriesDisplay({args.dsname: ds}, figsize=args.figsize)
        # print(f'TITLE: {args.title}')
        # yrange = getyrange(args.dsname, args.field)
        yrange = args.yrange

        display.plot(
            field=args.field, dsname=args.dsname, set_title=args.title, add_nan=args.add_nan,
            y_rng=yrange if yrange else None)
            # y_rng=list(map(int, args.set_yrange)) if args.set_yrange else None)
        # set_title=args.set_title, add_nan=args.add_nan)
        # day_night_background=args.day_night, **args.kwargs)
        # display.frameon(False)
        # display.title(None)
        # display.label(None)
        plt.axis(args.show_axis)
        plt.savefig(args.out_path, bbox_inches='tight')
        # plt.show(display.fig)
        plt.close(display.fig)

    if args.barbs_spd_dir:
        display.plot_barbs_from_spd_dir(
            dir_field=args.dir_field, spd_field=args.spd_field,
            pres_field=args.p_field, dsname=args.dsname,
            invert_y_axis=args.invert_y_axis, **args.kwargs)
        plt.savefig(args.out_path)
        # plt.show(display.fig)
        plt.close(display.fig)

    if args.barbs_u_v:
        display.plot_barbs_from_u_v(
            u_field=args.u_wind, v_field=args.v_wind, pres_field=args.p_field,
            dsname=args.dsname, set_title=args.set_title,
            invert_y_axis=args.invert_y_axis,
            day_night_background=args.day_night, num_barbs_x=args.num_barb_x,
            num_barbs_y=args.num_barb_y, **args.kwargs)
        plt.savefig(args.out_path)
        # plt.show(display.fig)
        plt.close(display.fig)

    if args.xsection_from_1d:
        display.plot_time_height_xsection_from_1d_data(
            data_field=args.field, pres_field=args.p_field, dsname=args.dsname,
            set_title=args.set_title, day_night_background=args.day_night,
            num_time_periods=args.num_time_periods, num_y_levels=args.num_y_levels,
            invert_y_axis=args.invert_y_axis, **args.kwargs)
        plt.savefig(args.out_path)
        # plt.show(display.fig)
        plt.close(display.fig)

    if args.time_height_scatter:
        display.time_height_scatter(
            data_field=args.field, dsname=args.dsname,
            cmap=args.cmap, alt_label=args.alt_label,
            alt_field=args.alt_field, cb_label=args.cb_label,
            **args.kwargs)
        plt.savefig(args.out_path)
        # plt.show(display.fig)
        plt.close(display.fig)

    ds.close()


def histogram(args):
    ds = act.io.armfiles.read_netcdf(args.file_path)

    display = act.plotting.HistogramDisplay({args.dsname: ds}, figsize=args.figsize)

    if args.stacked_bar_graph:
        display.plot_stacked_bar_graph(
            field=args.field, dsname=args.dsname, bins=args.bins,
            sortby_field=args.sortby_field, sortby_bins=args.sortby_bins,
            set_title=args.set_title, density=args.density, **args.kwargs)
        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    if args.size_dist:
        display.plot_size_distribution(
            field=args.field, bins=args.bins, time=args.time,
            dsname=args.dsname, set_title=args.set_title, **args.kwargs)
        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    if args.stairstep:
        display.plot_stairstep_graph(
            field=args.field, dsname=args.dsname, bins=args.bins,
            sortby_field=args.sortby_field, sortby_bins=args.sortby_bins,
            plot_quartiles=args.plot_quartiles, set_title=args.set_title,
            density=args.density, **args.kwargs)
        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    if args.heatmap:
        display.plot_heatmap(
            x_field=args.x_field, y_field=args.y_field, dsname=args.dsname,
            x_bins=args.x_bins, y_bins=args.y_bins, set_title=args.set_title,
            plot_quartiles=args.plot_quartiles, density=args.density,
            **args.kwargs)
        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    ds.close()


def contour(args):
    files = glob.glob(args.file_path)
    files.sort()

    time = args.time
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = act.io.armfiles.read_netcdf(f)
        data.update({f: ds})
        fields.update({f: args.fields})
        wind_fields.update({f: args.wind_fields})
        station_fields.update({f: args.station_fields})

    display = act.plotting.ContourDisplay(data, figsize=args.figsize)

    if args.create_contour:
        display.create_contour(fields=fields, time=time, function=args.function,
                               grid_delta=args.grid_delta,
                               grid_buffer=args.grid_buffer,
                               cmap=pyart.graph.cm_colorblind.HomeyerRainbow,
                               **args.kwargs)
    if args.contourf:
        display.contourf(x=args.x, y=args.y, z=args.z)
    if args.plot_contour:
        display.contour(x=args.x, y=args.y, z=args.z)
    if args.vectors_spd_dir:
        display.plot_vectors_from_spd_dir(fields=wind_fields, time=time,
                                          mesh=args.mesh, function=args.function,
                                          grid_delta=args.grid_delta,
                                          grid_buffer=args.grid_buffer)

    if args.barbs:
        display.barbs(x=args.x, y=args.y, u=args.u, v=args.v)
    if args.plot_station:
        display.plot_station(fields=station_fields, time=time,
                             text_color=args.text_color)

    plt.savefig(args.out_path)
    plt.show(display.fig)
    plt.close(display.fig)

    ds.close()


def geodisplay(args):
    ds = act.io.armfiles.read_netcdf(args.file_path)

    display = act.plotting.GeographicPlotDisplay({args.dsname: ds},
                                                 figsize=args.figsize)

    display.geoplot(data_field=args.field, lat_field=args.latitude,
                    lon_field=args.longitude, dsname=args.dsname,
                    cbar_label=args.cb_label, title=args.set_title,
                    plot_buffer=args.plot_buffer, stamen=args.stamen,
                    tile=args.tile, cartopy_feature=args.cfeatures,
                    cmap=args.cmap, text=args.text, gridlines=args.gridlines,
                    **args.kwargs)
    plt.savefig(args.out_path)
    plt.show(display.fig)
    plt.close(display.fig)

    ds.close()


def skewt(args):
    ds = act.io.armfiles.read_netcdf(args.file_path)

    display = act.plotting.SkewTDisplay({args.dsname: ds}, figsize=args.figsize)

    if args.from_u_and_v:
        display.plot_from_u_and_v(u_field=args.u_wind, v_field=args.v_wind,
                                  p_field=args.p_field, t_field=args.t_field,
                                  td_field=args.td_field, **args.kwargs)
        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    if args.from_spd_and_dir:
        display.plot_from_spd_and_dir(spd_field=args.spd_field,
                                      dir_field=args.dir_field,
                                      p_field=args.p_field,
                                      t_field=args.t_field,
                                      td_field=args.td_field,
                                      **args.kwargs)
        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    ds.close()


def xsection(args):
    ds = act.io.armfiles.read_netcdf(args.file_path)

    display = act.plotting.XSectionDisplay({args.daname: ds}, figsize=args.figsize)

    if args.plot_xsection:
        display.plot_xsection(dsname=args.dsname, varname=args.varname,
                              x=args.x_field, y=args.y_field,
                              sel_kwargs=args.sel_kwargs,
                              isel_kwargs=args.isel_kwargs, **args.kwargs)

        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    if args.xsection_map:
        display.plot_xsection_map(dsname=args.dsname, varname=args.varname,
                                  x=args.x_field, y=args.y_field,
                                  coastlines=args.coastlines,
                                  background=args.background,
                                  sel_kwargs=args.sel_kwargs,
                                  isel_kwargs=args.isel_kwargs,
                                  **args.kwargs)
        plt.savefig(args.out_path)
        plt.show(display.fig)
        plt.close(display.fig)

    ds.close()


def wind_rose(args):
    ds = act.io.armfiles.read_netcdf(args.file_path)

    display = act.plotting.WindRoseDisplay({args.dsname: ds}, figsize=args.figsize)

    display.plot(dir_field=args.dir_field, spd_field=args.spd_field,
                 dsname=args.dsname, cmap=args.cmap, set_title=args.set_title,
                 num_dirs=args.num_dir, spd_bins=args.spd_bins,
                 tick_interval=args.tick_interval, **args.kwargs)
    plt.savefig(args.out_path)
    plt.show(display.fig)
    plt.close(display.fig)

    ds.close()


def mtimeseries(args):
    outPath = args.out_path
    outFile = os.path.basename(outPath)
    splitParts = outFile.split('.')
    finalOutFile = ''
    for idx in range(0, len(splitParts) - 2):
        finalOutFile += splitParts[idx] + '.'

    finalOutFile += 'png'
    finalOutPath = outPath.replace(outFile, finalOutFile)
    finalOutPath = finalOutPath.replace('/.icons', '')
    print(finalOutPath)
    args.out_path = finalOutPath
    filePaths = args.file_paths

    print(filePaths)
    print(args.dsname)

    if os.path.exists(args.out_path):
        return
    try:
        ds = act.io.armfiles.read_netcdf(args.file_path)
        display = act.plotting.TimeSeriesDisplay({args.dsname: ds}, subplot_shape=(len(args.pm_list),),
                                                 #                                         figsize=(10.0, 15.0))
                                                 figsize=(len(args.pm_list), len(args.pm_list) * 2.5))

        numPM = len(args.pm_list)
        gridSize = math.ceil(numPM / 2.0)
        numSubplots = math.ceil(numPM / 2.0)
        idx = 0
        rowIdx = 1
        colIdx = 1
        for pm in args.pm_list:
            spIdx = rowIdx + (rowIdx % 3)
            print(pm)
            display.plot(pm, args.dsname, subplot_index=(idx,))
            idx += 1

        plt.subplots_adjust(hspace=0.5)
        plt.savefig(args.out_path, bbox_inches='tight')
        plt.close(display.fig)
    except:
        print("Failed to plot: " + args.file_path)
        fileOut = open("failed_to_plot.txt", "a+")
        fileOut.write(args.file_path + "\n")
        fileOut.close()
        plt.close()
        return

    '''
    display = act.plotting.TimeSeriesDisplay(args.dsname, subplot_shape=(len(filePaths),), figsize=(7.4, 9.0))
    for filePath in filePaths:

        ds = act.io.armfiles.read_netcdf(filePath)
        args.file_path = filePath
        display.plot(
            field=args.field, dsname=args.dsname, cmap=args.cmap,
            set_title=args.set_title, add_nan=args.add_nan)
            #day_night_background=args.day_night, **args.kwargs)

    plt.savefig(args.out_path)
    #plt.show(display.fig)
    plt.close(display.fig)
    '''
    return
