#!/usr/bin/env python

import argparse

import PIL
import numpy as np
import act
import pyart
import matplotlib.pyplot as plt
from datetime import datetime
import json
import glob
import cartopy
import astral
import os
import csv
import math
import sys
from PIL import Image
import multiprocessing
import time

import crontab

from pathlib import Path

import psycopg2
from settings import DB_SETTINGS


def getDBConnection():
    dbname = "dbname='" + DB_SETTINGS['dbname'] + "' "
    user = "user='" + DB_SETTINGS['user'] + "' "
    host = "host='" + DB_SETTINGS['host'] + "' "
    password = "password='" + DB_SETTINGS['password'] + "' "
    port = "port=" + DB_SETTINGS['port'] + " "
    connection_timeout = "connect_timeout=1800" + " "
    prepare_threshold = "prepare_threshold=0" + " "

    dbString = dbname + user + host + password + port + connection_timeout
    print(dbString)

    dbConnection = None
    try:
        dbConnection = psycopg2.connect(dbString)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return None
    finally:
        if dbConnection is not None:
            return dbConnection


def get_file_date(filePath):
    strDate = os.path.basename(os.path.normpath(filePath)).split('.')[2]
    return datetime.strptime(strDate, '%Y%m%d')


def getDateStr(dateObj):
    return datetime.strftime(dateObj, '%Y%m%d')


def getStartDate(dates, selectedDate):
    return min(dates, key=lambda currDate: abs(currDate - selectedDate))


def getPrimaryMeasurements(statement, dbCursor):
    dbCursor.execute(statement)

    resultList = []
    for result in dbCursor:
        resultList.append(list(result)[0])
        print(list(result)[0])
    return resultList


def getPathStrs(dataDir):
    pathlistNC = Path(dataDir).glob('**/*.nc')
    pathlistCDF = Path(dataDir).glob('**/*.cdf')
    pathStrs = list(map(str, pathlistNC))
    pathStrs += list(map(str, pathlistCDF))

    return pathStrs


def getSortedFileIndices(startDate, dateOffset, pathStrs):
    dates = []
    try:
        dates = list(map(get_file_date, pathStrs))
    except:
        print("badDate")
        return
    npDates = np.array(dates)
    npDatesSortedIdxs = np.argsort(npDates)

    currentIdxs = []
    if startDate == 'current':
        startIdx = -1 if int(dateOffset) == 0 else -1 * int(dateOffset)
        currentIdxs = npDatesSortedIdxs[startIdx:]
    else:
        dateStrs = list(map(getDateStr, dates))
        startDate = datetime.strptime(startDate, '%Y%m%d')
        selectedDate = getStartDate(dates, startDate)
        idx = dateStrs.index(datetime.strftime(selectedDate, "%Y%m%d"))

        sortedDateIdx = np.argwhere(npDatesSortedIdxs == idx)
        sortedDateIdx = np.asscalar(sortedDateIdx)
        idxStart = max(sortedDateIdx - int(dateOffset), 0)

        currentIdxs = npDatesSortedIdxs[idxStart:sortedDateIdx + 1]

    return currentIdxs


def getOutputFilePath(siteName, dataStreamName, baseOutDir, outDir, figSizeDir, pmResult, dataFilePath):
    dataFname = os.path.basename(dataFilePath)
    splitDFname = dataFname.split('.')
    dFStr = datetime.strptime(splitDFname[2], '%Y%m%d')

    monthStr = str(dFStr.month)
    if len(monthStr) == 1:
        monthStr = "0" + monthStr

    yearMonth = str(dFStr.year) + monthStr
    dateDir = splitDFname[0] + '.' + splitDFname[1] + '.' + yearMonth

    finalOutputDir = baseOutDir + str(dFStr.year) + '/' + siteName + '/' + dataStreamName + '/' + dateDir
    print(finalOutputDir)
    if not os.path.exists(finalOutputDir):
        os.makedirs(finalOutputDir)
    if not os.path.exists(finalOutputDir + '/.icons'):
        os.makedirs(finalOutputDir + '/.icons')

    inFilePrefix = str(os.path.basename(os.path.normpath(dataFilePath)))
    outFilePrefix = inFilePrefix.replace(".nc", ".")
    outFilePrefix = outFilePrefix.replace(".cdf", ".")
    outFile = outDir + '/' + outFilePrefix
    outFilePrefix = figSizeDir + '/' + outFilePrefix

    outPath = finalOutputDir + outFilePrefix + pmResult + '.png'
    return outPath


def getSegmentName(dataFilePath):
    dataFname = os.path.basename(dataFilePath)
    splitDFname = dataFname.split('.')
    dFStr = datetime.strptime(splitDFname[2], '%Y%m%d')

    monthStr = str(dFStr.month)
    if len(monthStr) == 1:
        monthStr = "0" + monthStr

    dayStr = str(dFStr.day)
    if len(dayStr) == 1:
        dayStr = "0" + dayStr

    dayTime = splitDFname[3]

    yearMonth = str(dFStr.year) + monthStr + dayStr + '.' + dayTime
    dateDir = splitDFname[0] + '.' + splitDFname[1] + '.' + yearMonth

    return dateDir


def getPlotFilePath(siteName, dataStreamName, baseOutDir, outDir, figSizeDir, dataFilePath):
    dataFname = os.path.basename(dataFilePath)
    splitDFname = dataFname.split('.')
    dFStr = datetime.strptime(splitDFname[2], '%Y%m%d')

    monthStr = str(dFStr.month)
    if len(monthStr) == 1:
        monthStr = "0" + monthStr

    yearMonth = str(dFStr.year) + monthStr
    dateDir = splitDFname[0] + '.' + splitDFname[1] + '.' + yearMonth

    finalOutputDir = baseOutDir + str(dFStr.year) + '/' + siteName + '/' + dataStreamName + '/' + dateDir
    print(finalOutputDir)
    if not os.path.exists(finalOutputDir):
        os.makedirs(finalOutputDir)
    if not os.path.exists(finalOutputDir + '/.icons'):
        os.makedirs(finalOutputDir + '/.icons')

    inFilePrefix = str(os.path.basename(os.path.normpath(dataFilePath)))
    outFilePrefix = inFilePrefix.replace(".nc", "")
    outFilePrefix = outFilePrefix.replace(".cdf", "")
    outFile = outDir + '/' + outFilePrefix
    outFilePrefix = figSizeDir + '/' + outFilePrefix

    outPath = finalOutputDir + outFilePrefix + '.png'
    return outPath


def combineImages(imagePaths, plot_file_path):
    try:

        images = [Image.open(x) for x in imagePaths]
        widths, heights = zip(*(i.size for i in images))

        max_height = max(heights)
        max_width = max(widths)
        total_width = max_width * len(imagePaths)
        total_height = max_height * len(imagePaths)

        new_im = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += max_height
        '''
        new_im = Image.new('RGB', (int(total_width/2), int(max_height*2)), (255, 255, 255))

        x_offset = 0
        y_offset = 0
        is_half = False
        count = 1
        for im in images:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += max_width
            if count == math.ceil((len(imagePaths) / 2.0)):
                x_offset = 0
                y_offset += max_height
            count += 1
        '''
        new_im.save(plot_file_path)
    except Exception as e:
        print('=-========= FAILED TO COMBINE IMAGES ======================')
        print(e)
        plt.close()


def insert_row(datastream, var_name, thumbnail_url, plot_url, base_out_dir, db_connection):
    thumbnail_url = thumbnail_url.replace(base_out_dir, 'https://www.archive.arm.gov/quicklooks/')
    plot_url = plot_url.replace(base_out_dir, 'https://www.archive.arm.gov/quicklooks/')

    stmt = 'INSERT into arm_int2.datastream_plot_info VALUES (%s, %s, %s, %s)'
    print(stmt)
    print('DSP: ' + datastream)
    try:
        db_cursor = db_connection.cursor()
        # db_cursor.execute(stmt, (datastream, var_name, thumbnail_url, plot_url))
        # db_connection.commit()
        db_cursor.close()
    except Exception as e:
        print(e)
        #update_row(datastream, var_name, thumbnail_url, db_connection)


def update_row(datastream, var_name, thumbnail_url, db_connection):
    UPDATE_PRIMARY_MEASUREMENTS = 'update arm_int2.datastream_var_name_info set thumbnail_url = %s where ' \
                                  'datastream = %s and var_name = %s '
    print(UPDATE_PRIMARY_MEASUREMENTS)
    print('DSP: ' + datastream)
    try:
        db_cursor = db_connection.cursor()
        db_cursor.execute(UPDATE_PRIMARY_MEASUREMENTS, (datastream, var_name, thumbnail_url))
        # db_connection.commit()
        db_cursor.close()
    except Exception as e:
        print(e)


def getPrimaryForDatastream(dbCursor, data_stream_name, statement, args, date_offset, base_out_dir):
    site_name = data_stream_name[0:3]

    print("*****************************************************************************")
    print("Current input directory: " + args.data_dir)
    print("*****************************************************************************\n")
    print("Creating plots for the following variables...\n")

    result_list = getPrimaryMeasurements(statement, dbCursor)
    if len(result_list) == 0:
        file_out_pm = open("/tmp/no_pm.txt", "a+")
        file_out_pm.write(args.data_dir + "\n")
        file_out_pm.close()
        return

    out_dir = args.out_dir
    path_strs = getPathStrs(args.data_dir)
    current_idxs = getSortedFileIndices(args.start_date, date_offset, path_strs)

    args.file_paths = []
    args.dsname = data_stream_name

    print("\nCurrent output directory: " + out_dir + "\n")
    for current_idx in current_idxs:
        path_in_str = path_strs[current_idx]

        args.file_path = path_in_str
        print("Current input file: " + path_in_str + "\n")
        print("Creating output files...\n")

        fig_sizes = [(1.0, 1.0), (7.4, 4.0)]
        fig_size_dirs = ['/.icons', '']

        image_paths = []
        plot_file_path = getPlotFilePath(site_name, data_stream_name, base_out_dir, out_dir, "", path_in_str)
        for idx in range(0, 2):
            for result in result_list:
                print(args.out_path)

                args.field = result

                fig_size_dir = fig_size_dirs[idx]

                args.out_path = getOutputFilePath(site_name, data_stream_name, base_out_dir, out_dir, fig_size_dir,
                                                  result, path_in_str)
                if idx == 1:
                    image_paths.append(args.out_path)

                args.figsize = fig_sizes[idx]

                print(args.out_path)

                insert_row(data_stream_name, result, args.out_path, plot_file_path)
                # if (os.path.exists(args.out_path)):
                #    continue
                try:
                    args.set_title = getSegmentName(path_in_str) + ": " + result
                    args.action(args)
                except:
                    print("Failed to process: " + path_in_str)
                    file_out = open("/tmp/bad_datastreams.txt", "a+")
                    file_out.write(args.out_path + "\n")
                    file_out.close()
                    plt.close()
                    return
            plt.close()
        if len(image_paths) > 0:
            try:
                combineImages(image_paths, plot_file_path)
                # plot_file_path = getPlotFilePath(site_name, data_stream_name, base_out_dir, out_dir, "", path_in_str)
                # if not (os.path.exists(plot_file_path)):
                #    combineImages(image_paths, plot_file_path)
            except Exception as e:
                plt.close()
                print("FAILED TO WRITE IMG: " + str(e))
        print("\n...done\n")
        '''
        varDict['pm_list'] = resultList
        try:
            mtimeseries(args)
        except:
            print("Failed to process: " + path_in_str)
            fileOut = open("failed_multi.txt", "a+")
            fileOut.write(varDict['out_path'] + "\n")
            fileOut.close()
            plt.close()
            return
        '''
    print("*****************************************************************************\n\n\n")


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
        # print(ds)
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


def timeseries(args):
    ds = act.io.armfiles.read_netcdf(args.file_path)
    if args.plot:
        display = act.plotting.TimeSeriesDisplay({args.dsname: ds}, figsize=args.figsize)
        print("TITLE: " + args.title)
        display.plot(
            field=args.field, dsname=args.dsname, set_title=args.title, add_nan=args.add_nan)
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


def parseStartDate(startDate):
    try:
        return datetime.strptime(startDate, '%Y%m%d')
    except:
        if startDate != 'current':
            raise ValueError("Incorrect date format. Should be 'YYYYMMDD'. Defaulting to latest data.")
        return 'current'


def parseCsv(csvName, baseDir, baseOutDir):
    config = []
    with open(csvName, newline='') as configCsv:

        configRows = csv.DictReader(configCsv)
        for row in configRows:

            outDir = baseOutDir + row['datastream'][:3] + '/' + row['datastream']
            dataDir = baseDir + row['datastream'][:3] + '/' + row['datastream']

            datastream = dict()
            datastream['dir'] = dataDir
            datastream['outDir'] = outDir
            datastream['name'] = row['datastream']
            datastream['range_offset'] = row['range_offset']

            startDate = parseStartDate(row['start_date'])
            if (type(startDate) is datetime):
                datastream['startDate'] = startDate.strftime('%Y%m%d')
            else:
                datastream['startDate'] = 'current'

            # print(datastream)

            config.append(datastream)
    return config


def processBulk(args):
    varDict = vars(args)
    dsDir = args.ds_dir

    SELECT_PRIMARY_MEASUREMENTS = "select d.var_name from datastream_var_name_info d where d.datastream = "

    dbConnection = getDBConnection()
    dbCursor = None
    try:
        dbCursor = dbConnection.cursor()
    except:
        time.sleep(3)
        dbConnection = getDBConnection()
        dbCursor = dbConnection.cursor()
    args.data_dir = dsDir
    args.out_dir = args.base_out_dir + os.path.basename(dsDir)
    # varDict['out_dir'] = varDict['base_out_dir'] + os.path.basename(dsDir)
    dataType = os.path.basename(dsDir)
    args.start_date = 'current'
    # varDict['start_date'] = 'current'
    currentSelect = SELECT_PRIMARY_MEASUREMENTS + "'%s'" % (dataType)
    getPrimaryForDatastream(dbCursor, dataType, currentSelect, args, 0,
                            args.base_out_dir)
    # getPrimaryForDatastream(dbCursor, dataType, currentSelect, varDict, args, 5,
    #                        varDict['base_out_dir'])
    dbConnection.close()


def processSubset(args):
    SELECT_PRIMARY_MEASUREMENTS = "select d.var_name from datastream_var_name_info d where d.datastream = "

    datastream = args.datastream
    dbConnection = getDBConnection()
    dbCursor = dbConnection.cursor()
    args.data_dir = datastream['dir']
    args.out_dir = datastream['outDir']
    dataType = datastream['name']
    args.start_date = datastream['startDate']
    currentSelect = SELECT_PRIMARY_MEASUREMENTS + "'%s'" % (dataType)
    getPrimaryForDatastream(dbCursor, dataType, currentSelect, args, datastream['range_offset'],
                            args.base_out_dir)
    dbConnection.close()


def getOutputUrl(siteName, dataStreamName, baseOutDir, outDir, figSizeDir, pmResult, dataFilePath):
    dataFname = os.path.basename(dataFilePath)
    splitDFname = dataFname.split('.')
    dFStr = datetime.strptime(splitDFname[2], '%Y%m%d')

    monthStr = str(dFStr.month)
    if len(monthStr) == 1:
        monthStr = "0" + monthStr
    yearMonth = str(dFStr.year) + monthStr
    dateDir = splitDFname[0] + '.' + splitDFname[1] + '.' + yearMonth

    urlStr = 'https://www.archive.arm.gov/quicklooks/'
    finalOutputDir = urlStr + str(dFStr.year) + '/' + siteName + '/' + dataStreamName + '/' + dateDir

    inFilePrefix = str(os.path.basename(os.path.normpath(dataFilePath)))
    outFilePrefix = inFilePrefix.replace(".nc", ".")
    outFilePrefix = outFilePrefix.replace(".cdf", ".")
    outFile = outDir + '/' + outFilePrefix
    outFilePrefix = figSizeDir + '/' + outFilePrefix

    outPath = finalOutputDir + outFilePrefix + pmResult + '.png'
    return outPath

def createGiriInsert(dataStreamName, varName):
    startDate = '2019-11-01 00:00:00'
    endDate   = '9999-09-09 00:00:00'
    rowEntry  = dataStreamName + '|' + varName + '|' + varName + '|' + '|' + startDate + '|' + endDate
    return rowEntry

def createPreSelectInsert(dataStreamName, varName, urlStr):
    startDate = '2019-11-01 00:00:00'
    endDate   = '2019-12-15 00:00:00'
    rowEntry  = dataStreamName + '|' + varName + '|' + varName + '|' + startDate + '|' + endDate + '|' + urlStr
    return rowEntry

def get_data_stream_name(path):
    return str(os.path.basename(os.path.normpath(path)))

def remove_raw_DS(ds_names):
    return ds_names.find(".a1") == -1



# def getPrimaryForDs(data_stream_name, args, date_offset, base_out_dir, result_list):
def getPrimaryForDs(args):
    data_stream_name = args.dsname
    date_offset = args.date_offset
    base_out_dir = args.base_out_dir
    result_list = args.pm_list

    args.data_dir = args.ds_dir
    args.out_dir = args.base_out_dir + os.path.basename(args.ds_dir)

    site_name = data_stream_name[0:3]

    print("*****************************************************************************")
    print("Current input directory: " + args.data_dir)
    print("*****************************************************************************\n")
    print("Creating plots for the following variables...\n")

    out_dir = args.out_dir
    path_strs = getPathStrs(args.data_dir)
    current_idxs = getSortedFileIndices(args.start_date, date_offset, path_strs)

    args.file_paths = []
    args.dsname = data_stream_name

    print("\nCurrent output directory: " + out_dir + "\n")
    for current_idx in current_idxs:

        # db_connection = ''
        db_connection = getDBConnection()
        db_connection.set_session(autocommit=True)

        path_in_str = path_strs[current_idx]
        args.file_path = path_in_str
        if os.path.getsize(path_in_str) > args.max_file_size: # exclude if file size is > 100MB
            print('File too large:', path_in_str, ' : ', os.path.getsize(path_in_str))
            continue
        # if True: continue
        print("Current input file: " + path_in_str + "\n")
        print("Creating output files...\n")

        fig_sizes = [(1.0, 1.0), (7.4, 4.0)]
        fig_size_dirs = ['/.icons', '']

        row_components = []
        image_paths = []
        plot_file_path = getPlotFilePath(site_name, data_stream_name, base_out_dir, out_dir, "", path_in_str)
        for idx in range(0, 2):
            for result in result_list:
                print(args.out_path)

                args.field = result
                fig_size_dir = fig_size_dirs[idx]
                args.out_path = getOutputFilePath(site_name, data_stream_name, base_out_dir, out_dir, fig_size_dir,
                                                  result, path_in_str)
                if idx == 1:
                    image_paths.append(args.out_path)

                args.figsize = fig_sizes[idx]
                print(args.out_path)

                if os.path.exists(args.out_path):
                    urlStr = getOutputUrl(site_name, data_stream_name, base_out_dir, out_dir, fig_size_dir, result, path_in_str)
                    rowEntry = createPreSelectInsert(data_stream_name, result, urlStr)
                    rowEntryB = createGiriInsert(data_stream_name, result)
                    print('URL STRING: '  + urlStr)
                    print('ROWA STRING: '  + rowEntry)
                    print('ROWB STRING: '  + rowEntryB)
                    #    insert_row(getSegmentName(path_in_str), result, args.out_path, plot_file_path, base_out_dir,
                    #               db_connection)
                    continue
                try:
                    if idx == 1:
                        args.title = getSegmentName(path_in_str) + " " + result
                        args.show_axis = 'on'
                    else:
                        args.title = ""
                        args.show_axis = 'off'
                        row_component = {'segment_name': getSegmentName(path_in_str), 'var_name': result,
                                         'out_path': args.out_path, 'plot_path': plot_file_path}
                        row_components.append(row_component)
                    args.action(args)
                except Exception as e:
                    print("FAILED PROCESS: " + str(e))
                    print("Failed to process: " + path_in_str)
                    file_out = open("/tmp/bad_datastreams.txt", "a+")
                    file_out.write(args.out_path + "\n")
                    file_out.close()
                    try:
                        plt.close()
                    except: print('=-=-=- Failed to close "plt" -=-=-=')
                    db_connection.close()
                    return
            try: plt.close()
            except: print('=-=-=- Failed to close "plt", AGAIN -=-=-=')

        if len(image_paths) > 0:
            try:
                combineImages(image_paths, plot_file_path)
                plot_file_path = getPlotFilePath(site_name, data_stream_name, base_out_dir, out_dir, "", path_in_str)
                print(plot_file_path)
                if not (os.path.exists(plot_file_path)):
                    combineImages(image_paths, plot_file_path)
                for row in row_components:
                    insert_row(row['segment_name'], row['var_name'], row['out_path'], row['plot_path'], base_out_dir,
                              db_connection)
            except Exception as e:
                try: plt.close()
                except: print('=-=-=- Failed to close "plt" FOR THE THIRD TIME -=-=-=')

                print("FAILED TO WRITE IMG: " + str(e))
                db_connection.close()
        print("\n...done\n")
        db_connection.close()
        '''
        varDict['pm_list'] = resultList
        try:
            mtimeseries(args)
        except:
            print("Failed to process: " + path_in_str)
            fileOut = open("failed_multi.txt", "a+")
            fileOut.write(varDict['out_path'] + "\n")
            fileOut.close()
            plt.close()
            return
        '''
    print("*****************************************************************************\n\n\n")


def getArgs():
    parser = argparse.ArgumentParser(description='Create GeoDisplay Plot')
    parser.add_argument('-days', '--num_days', type=str,
                        help='number of days offset')
    parser.add_argument('-nt', '--num_t', type=str,
                        help='Max number of threads')
    parser.add_argument('-sites', '--site_list', type=str,
                        help='comma separated list of sites')
    # parser.add_argument('-sitestxt', '--site_txt', type=str,
    #                     help='comma separated list of sites')
    parser.add_argument('-useTxtFile', '--use-txt-file', dest='use_txt_file', action='store_true', default=False,
                        help='''Indicates to use a .txt file in the same directory which contains a 
                        static list of datastreams to process. Used in conjunction with the -sites argument, 
                        the name of the sites will be the relative name for the text file. 
                        Example: "-sites anx --use-txt-file", a file named "anx.txt" must exist within 
                        the same directory and contain newline-separated names of datastreams.''')
    parser.add_argument('-maxFs', '--max-file-size', type=int, default=100000000, dest='max_file_size',
                        help='max file size in number of bytes - default is 100000000 (100MB)')
    # parser.add_argument('-q', '--quiet', action='store_false',
    #                     help='silence a lot of the logging output')

    parser.add_argument('-cfg', '--config', type=str,
                        help='Config file to use for creating Plot')
    parser.add_argument('-base', '--base_dir', type=str, default='/data/archive/',
                        help='Base Directory to use for creating Plot')
    parser.add_argument('-baseOut', '--base_out_dir', type=str, default='/data/ql_plots/',
                        help='Base Out Directory to use for saving Plot')
    parser.add_argument('-dd', '--data_dir', type=str,
                        help='File to use for creating Plot')
    parser.add_argument('-f', '--file_path', type=str,
                        help='File to use for creating Plot')
    parser.add_argument('-o', '--out_path', type=str,
                        help='File path to use for saving image')
    parser.add_argument('-fd', '--field', type=str, default=None,
                        help='Name of the field to plot')
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
    parser.add_argument('-p', '--plot', default=False, action='store_true',
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
    parser.add_argument('-ts', '--timeseries', dest='action',
                        action='store_const', const=timeseries)
    parser.add_argument('-mts', '--mtimeseries', dest='action',
                        action='store_const', const=mtimeseries)
    parser.add_argument('-c', '--contour', dest='action',
                        action='store_const', const=contour)
    parser.add_argument('-hs', '--histogram', dest='action',
                        action='store_const', const=histogram)
    parser.add_argument('-ex', '--exclude', type=str, default='',
                        help='Comma separated list. Exclude any datastreams containing these values in their name.')
    return parser.parse_args()


# def getDsListFromTxt(sites):
#     txt_files = {}
#     for site in sites:
#         site_datastreams_file = site+'.txt'
#         if os.path.isfile(site_datastreams_file):
#             with open(site_datastreams_file, 'r') as site_datastreams:
#                 txt_files[site] = site_datastreams.read()
#         else: print('!![ERROR]!! Failed to read site txt file: ', site_datastreams_file)
#     return txt_files

def readDatastreamsFromSiteTxt(site):
    site_datastreams_file = site+'.txt'
    if os.path.isfile(site_datastreams_file):
        with open(site_datastreams_file, 'r') as site_datastreams:
            return site_datastreams.readlines()
    else: print('!![ERROR]!! Failed to read site txt file: ', site_datastreams_file)


# def buildDsPathsFromTxt(sites):
#     paths = {}
#     for site in sites:
#         site_datastreams = readDatastreamsFromSiteTxt(site)
            # dsnames = site_datastreams.read()
        # dspaths = [os.path.join('/data/archive/', site, ds.strip()) for ds in site_datastreams]
        # paths[site] = dspaths
    # print('paths:', paths)
    # return paths

def buildDsPaths(site, dsnames=None):
    site_datastreams = readDatastreamsFromSiteTxt(site) if dsnames is None else dsnames
    dspaths = [os.path.join('/data/archive/', site, ds.strip()) for ds in site_datastreams]
    print('dspaths:', dspaths)
    return dspaths


def oldwayGetDsNames(sites):
    def getSelectedSitePaths(sites):
        data_archive_path = '/data/archive/'
        data_archive_dirs = [os.path.join(data_archive_path, o) for o in os.listdir(data_archive_path) if
                             os.path.isdir(os.path.join(data_archive_path, o))]
        data_archive_dirs = list(filter(remove_raw_DS, data_archive_dirs))
        return [site for site in data_archive_dirs if os.path.basename(site) in sites]

    selected_sites = getSelectedSitePaths(sites)
    for site_path in selected_sites:
        print(str(os.path.basename(site_path)))
        site_path_dirs = [os.path.join(site_path, o) for o in os.listdir(site_path) if
                          os.path.isdir(os.path.join(site_path, o))]

        data_file_paths = list(map(str, site_path_dirs)) # /data/archive/site_code/datastream.lvl
        data_file_paths = list(filter(remove_raw_DS, data_file_paths))

        ds_names = list(map(get_data_stream_name, data_file_paths))
        ds_names = list(filter(remove_raw_DS, ds_names))
        # if args.use_txt_file and datastream_txt_files:
        #     ds_names = list(filter(lambda ds : ds in datastream_txt_files[ds[:3]], ds_names))
        print(ds_names)
        return ds_names, data_file_paths

def buildPrimaryMeasurementDict(ds_names):
    ds_dict = {}
    for ds in ds_names:
        ds_dict[ds] = []

    SELECT_PRIMARY_MEASUREMENTS = "select d.datastream, d.var_name from arm_int2.datastream_var_name_info d where d.datastream IN %s"
    UPDATE_PRIMARY_MEASUREMENTS = "update arm_int2.datastream_var_name_info set thumbnail_url = %s where datastream = %s and var_name = %s"
    dbConnection = getDBConnection()
    dbCursor = dbConnection.cursor()
    dbCursor.execute(SELECT_PRIMARY_MEASUREMENTS, (tuple(ds_names),))
    results = dbCursor.fetchall()
    # print(results)
    for pm in results:
        ds_dict[pm[0]].append(pm[1])
    # print(ds_dict)
    dbCursor.close()
    dbConnection.close()
    return ds_dict


def proceed(args):
    ds_names = args.ds_names
    data_file_paths = args.data_file_paths

    max_th = int(args.num_t)
    # num_days = int(args.num_days)
    # print('num_days:', num_days)

    ds_dict = buildPrimaryMeasurementDict(ds_names)

    args.start_date = 'current'
    processes = []
    # max_th = num_t
    count = 0
    for ds in ds_names:
        args.ds_dir = data_file_paths[ds_names.index(ds)]  # This is so confusing!
        print("***************")
        print(ds_names.index(ds))
        print(data_file_paths[ds_names.index(ds)])
        args.dsname = ds
        args.date_offset = args.num_days
        args.pm_list = ds_dict[ds]
        ds_process = multiprocessing.Process(target=getPrimaryForDs, args=(args,))
        processes.append(ds_process)
        ds_process.start()

        if count % max_th == 0:
            for t in processes:
                t.join()
        count += 1

def main():
    args = getArgs()

    # varDict = vars(args)
    sites = args.site_list.split(',')
    print('sites:', sites)

    exclude = args.exclude.split(',')
    if exclude[0]: print('Excluding:', exclude)

    # datastream_txt_files = {}
    # if args.use_txt_file: datastream_txt_files = getDsListFromTxt(sites)


    if args.use_txt_file:
        print('reading sites from txt')
        # datastream_txt_files = getDsListFromTxt(sites)
        # dspaths = buildDsPathsFromTxt(sites)
        for site in sites:
            args.ds_names = [ ds.strip() for ds in readDatastreamsFromSiteTxt(site) ]
            print('[args.ds_names]:', args.ds_names)
            args.data_file_paths = buildDsPaths(site=site, dsnames=args.ds_names)
            print('[args.ds_data_file_paths]:', args.data_file_paths)
            proceed(args)

    else:
        oldway = oldwayGetDsNames(sites)
        args.ds_name = oldway[0]
        args.data_file_paths = oldway[1]
        proceed(args)

        # SELECT_PRIMARY_MEASUREMENTS = "select d.var_name from datastream_var_name_info d where d.datastream = "


        '''

        processes = []
        for datastream_dir in data_file_paths:
            varDict['ds_dir'] = datastream_dir
            #print(datastream_dir)
            ds_process = multiprocessing.Process(target=processBulk, args=(args,))
            processes.append(ds_process)
            ds_process.start()

        for t in processes:
            t.join()
        return
    print("Done with all!")
    return

    SELECT_PRIMARY_MEASUREMENTS = "select d.var_name from datastream_var_name_info d where d.datastream = "
    config = parseCsv(varDict['config'], varDict['base_dir'], varDict['base_out_dir'])
    for datastream in config:
        args.datastream = datastream
        ds_process = multiprocessing.Process(target=processSubset, args=(args,))
        processes.append(ds_process)
        ds_process.start()

    for t in processes:
        t.join()
        '''



if __name__ == '__main__':
    main()
    print("Done with all!")
