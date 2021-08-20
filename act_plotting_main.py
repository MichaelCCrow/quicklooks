#!/usr/bin/env python

import argparse

import PIL
import numpy as np
import act
import pyart
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
from itertools import product, groupby
from loguru import logger as log
import time
import re
from functools import partial
import copy

import crontab

from pathlib import Path

import psycopg2
from settings import DB_SETTINGS

log.remove()
log.add('logs/act.log', level='INFO', enqueue=True, colorize=True,
        format='<g>{time:YYYY-MM-DD HH:mm:ss!UTC}</g> | <lvl>{level: >5}</lvl> | <lvl>{message}</lvl>')

def getDBConnection():
    dbname = "dbname='" + DB_SETTINGS['dbname'] + "' "
    user = "user='" + DB_SETTINGS['user'] + "' "
    host = "host='" + DB_SETTINGS['host'] + "' "
    password = "password='" + DB_SETTINGS['password'] + "' "
    port = "port=" + DB_SETTINGS['port'] + " "
    connection_timeout = "connect_timeout=1800" + " "
    prepare_threshold = "prepare_threshold=0" + " "

    dbString = dbname + user + host + password + port + connection_timeout
    dbConnection = None
    try:
        dbConnection = psycopg2.connect(dbString)
        # print('[OPENED]')
    except (Exception, psycopg2.DatabaseError) as error:
        log.error(f'[ERROR] Database connection failed. {error}')
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

def offsetDays(file, days=1):
    file = Path(file)
    offset = datetime.fromtimestamp(file.stat().st_mtime) + timedelta(days=int(days))
    return offset > datetime.today()

def getPathStrs(dataDir):
    pathlistNC = Path(dataDir).glob('**/*.nc')
    # pathlistNC = list(filter(lambda p: offsetDays(p, days), pathlistNC))
    pathlistCDF = Path(dataDir).glob('**/*.cdf')
    # pathlistCDF = list(filter(lambda p: offsetDays(p, days), pathlistCDF))

    pathStrs = list(map(str, pathlistNC))
    pathStrs += list(map(str, pathlistCDF))

    return pathStrs

def getSortedFileIndices(startDate, dateOffset, pathStrs):
    # print('[getSortedFileIndices]')
    dates = []
    try:
        dates = list(map(get_file_date, pathStrs))
    except Exception as e:
        log.error(f'badDate -- {e}')
        return dates
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
    # print(finalOutputDir)
    if not os.path.exists(finalOutputDir):
        try:
            os.makedirs(finalOutputDir)
        except FileExistsError:
            pass
    if not os.path.exists(finalOutputDir + '/.icons'):
        try:
            os.makedirs(finalOutputDir + '/.icons')
        except FileExistsError:
            pass

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
    log.info('[finalOutputDir] ', finalOutputDir)
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
    # print(f'[inFilePrefix][{inFilePrefix}] [outFilePrefix][{outFilePrefix}] [outPath][{outPath}]')
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
        log.error('=-========= FAILED TO COMBINE IMAGES ======================')
        log.error(e)
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

def getinstcode(dsname):
    instrument_regex = r'(?<=\w{3})\w*(?=[A-Z]\w*)'
    groups = re.search(instrument_regex, dsname)
    return groups.group() if groups else None

def getyrange(dsname, varname):
    inst = getinstcode(dsname)
    if not inst:
        log.error(f'[ERROR] Could not parse instrument code from datastream name [{dsname}]')
        return None
    # noinspection SqlNoDataSourceInspection
    q = f'''
    SELECT ymin, ymax
    FROM arm.qls_yvar_range
    WHERE instrument_code = '{inst}'
    AND var_name = '{varname}'
    '''
    conn = getDBConnection()
    if conn is None: return None
    try:
        with conn.cursor() as curs:
            curs.execute(q)
            results = curs.fetchall()
            yrange = [int(n) for n in results[0]] if results else None
            # print(f'[yrange] [{inst}] [{varname}]', yrange)
            return yrange
    finally:
        conn.close()
        # print('[CLOSED]')

def timeseries(args):
    ds = act.io.armfiles.read_netcdf(args.file_path)
    # if args.field not in ds.data_vars.keys():
    #     log.warning(f'[SKIPPING][measurement-not-in-cdf] [{args.field}] [{os.path.basename(args.file_path)}]')
    #     ds.close()
    #     return False

    if args.plot:
        # print('[timeseries][plot]')
        display = act.plotting.TimeSeriesDisplay({args.dsname: ds}, figsize=args.figsize)
        # print(f'TITLE: {args.title}')
        yrange = getyrange(args.dsname, args.field)

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
    # return True

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

def getOutputUrl(siteName, dataStreamName, baseOutDir, outDir, figSizeDir, pmResult, dataFilePath):
    dataFname = os.path.basename(dataFilePath)
    splitDFname = dataFname.split('.')
    dFStr = datetime.strptime(splitDFname[2], '%Y%m%d')

    monthStr = str(dFStr.month)
    if len(monthStr) == 1:
        monthStr = "0" + monthStr
    yearMonth = str(dFStr.year) + monthStr
    dateDir = splitDFname[0] + '.' + splitDFname[1] + '.' + yearMonth

    urlStr = 'https://adc.arm.gov/quicklooks/'
    finalOutputDir = urlStr + str(dFStr.year) + '/' + siteName + '/' + dataStreamName + '/' + dateDir

    inFilePrefix = str(os.path.basename(os.path.normpath(dataFilePath)))
    outFilePrefix = inFilePrefix.replace(".nc", ".")
    outFilePrefix = outFilePrefix.replace(".cdf", ".")
    outFile = outDir + '/' + outFilePrefix
    outFilePrefix = figSizeDir + '/' + outFilePrefix

    outPath = finalOutputDir + outFilePrefix + pmResult + '.png'
    return outPath

def createGiriInsert(datastreamName, varName, startDate='1990-01-01 00:00:00', endDate='9999-09-09 00:00:00'):
    log.trace(f'[insert][giri_inventory] {datastreamName} : {varName} :: {startDate} - {endDate}')
    # noinspection SqlNoDataSourceInspection
    q = f'''
    INSERT INTO color.giri_inventory 
      (datastream, var_name, ql_var_name, measurement, start_date, end_date)
    SELECT
      '{datastreamName}', 
      '{varName}', 
      '{varName}', 
      dsinfo.primary_measurement,
      '{startDate}', 
      '{endDate}'
    FROM arm_int2.datastream_var_name_info dsinfo
    WHERE dsinfo.var_name = '{varName}'
      AND dsinfo.datastream = '{datastreamName}'
    ON CONFLICT ON CONSTRAINT giri_inventory_pkey
    DO UPDATE SET end_date = '{endDate}'
    '''
    # print('[giri_inventory]', q)
    return q


def createPreSelectInsert(datastreamName, varName, urlStr, startDate='1990-01-01 00:00:00', endDate='9999-09-09 00:00:00'):
    log.trace(f'[insert][pre_selected_qls_info] {datastreamName} : {varName} :: {startDate} - {endDate}')
    # noinspection SqlNoDataSourceInspection
    q = f'''
    INSERT INTO arm_int2.pre_selected_qls_info 
      (datastream, var_name, ql_var_name, start_date, end_date, thumbnail_url)
    VALUES (
      TRIM('{datastreamName}'),
      '{varName}', 
      '{varName}', 
      '{startDate}', 
      '{endDate}',
      '{urlStr}'
    ) 
    ON CONFLICT ON CONSTRAINT pre_selected_qs_pk
    DO UPDATE SET end_date = '{endDate}'
    '''
    # print('[pre_selected_qls_info]', q)
    return q

def get_data_stream_name(path):
    return str(os.path.basename(os.path.normpath(path)))

def remove_raw_DS(dsnames):
    return dsnames.find('.a1') == -1 and dsnames.find('.a0') == -1 and dsnames.find('.00') == -1

def insert_qls_info(q, conn, table):
    try:
        with conn.cursor() as curs:
            curs.execute(q)
            log.trace(f'[{table}] INSERT success')
    except Exception as e:
        log.error(f'[FAILED INSERT][{table}] {q} - [REASON] {e}')
        pass

fig_sizes = [(1.0, 1.0), (7.4, 4.0)]
fig_size_dirs = ['/.icons', '']

def processPm(args, site_name, out_dir, dsname, data_file_path, pm):
    idx_started = datetime.now()

    try:
        check_started = datetime.now()
        dscdf = act.io.armfiles.read_netcdf(data_file_path)
        if pm not in dscdf.data_vars.keys():
            log.debug(f'[SKIPPING][measurement-not-in-cdf] [{pm}] [{os.path.basename(data_file_path)}]')
            dscdf.close()
            return
    except Exception as e:
        log.critical(f'[FAILED][checking for measurements in cdf file] {data_file_path} [REASON] {e}')
    finally:
        check_ended = datetime.now() - check_started
        log.trace(f'[time][check-if-measurement-in-cdf-file] [{check_ended}]')

    args.file_path = data_file_path
    fsize = os.path.getsize(data_file_path)
    if fsize > args.max_file_size: # exclude if file size is > 100MB
        log.warning(f'File too large: {data_file_path} : [{fsize}]')
        return

    imgpath = ''

    # TODO: Figure out how to remove the necessity for a loop
    for idx in range(2):
        args.field = pm  # required for timeseries plotting method
        fig_size_dir = fig_size_dirs[idx]
        args.out_path = getOutputFilePath(site_name, dsname, args.base_out_dir, out_dir, fig_size_dir,
                                          pm, data_file_path)
        args.figsize = fig_sizes[idx]

        if os.path.exists(args.out_path):
            urlStr = getOutputUrl(site_name, dsname, args.base_out_dir, out_dir, fig_size_dir, pm, data_file_path)
            log.trace(f'[URL STRING] {urlStr}')
            pre_selected_qls_info_insert_query = createPreSelectInsert(dsname, pm, urlStr, endDate=args.end_dates[dsname])
            giri_inventory_insert_query = createGiriInsert(dsname, pm, endDate=args.end_dates[dsname])
            conn = getDBConnection()
            try:
                with conn:
                    insert_qls_info(pre_selected_qls_info_insert_query, conn, 'pre_selected_qls_info')
                    insert_qls_info(giri_inventory_insert_query, conn, 'giri_inventory')
            finally:
                conn.close()
                # print('[CLOSED]')
            continue
        try:
            if idx == 1:
                args.title = getSegmentName(data_file_path) + " " + pm
                args.show_axis = 'on'
            else:
                args.title = ""
                args.show_axis = 'off'
            action_started = datetime.now()
            try:
                args.action(args)  # now executes any methods flagged from command line args
                if idx == 1:
                    imgpath = args.out_path
                log.info(f'[plot-generated] {args.out_path}')
            except Exception as e:
                errmsg = f'[FAILED][args.action] [{data_file_path}] [{pm}] -- [REASON] [{e}]'
                log.error(errmsg)
                with open('logs/bad_datastreams.txt', 'a+') as f:
                    print(errmsg, file=f)
            finally:
                action_time = datetime.now() - action_started
                log.trace(f'[time][ACTION] {action_time}')
        except Exception as e:
            log.error(f'[FAILED to PROCESS] {data_file_path} [REASON] {str(e)}')
            if os.access('/tmp/bad_datastreams.txt', os.W_OK):
                with open('/tmp/bad_datastreams.txt', 'a+') as file_out:
                    print(args.out_path, file=file_out)
            plt.close()
            return
    plt.close()

    idx_time = datetime.now() - idx_started
    log.trace(f'[time][process-pm] {idx_time}')
    log.complete()

    return imgpath

def combine(image_paths):
    if len(image_paths) <= 0:
        log.warning('[combine][len<=0]')
        return
    plot_file_path = re.sub('([a-z_]+)(?=\.png)', '', image_paths[0]).replace('..', '.')
    log.info(f'[combine][total][{len(image_paths)}] -> {plot_file_path}')
    # plot_file_path = getPlotFilePath(site_name, dsname, args.base_out_dir, out_dir, '', path_in_str)
    try:
        combineImages(image_paths, plot_file_path)
        if not (os.path.exists(plot_file_path)):
            errmsg = f'[FAILED][COMBINE] Did not create file in [{plot_file_path}]'
            log.error(errmsg)
            with open('logs/err/combine.err', 'a+') as f:
                print(errmsg, file=f)
            combineImages(image_paths, plot_file_path)
    except Exception as e:
        plt.close()
        log.critical(f'FAILED TO WRITE IMG: {str(e)}')

def getPrimaryForDs(args, dsname):
    log.info('\n**************************************')
    log.info(f'[PROCESSING][datastream] [{dsname}]')
    args.dsname = dsname # required for other methods to find the dsname
    ds_dir = args.data_file_paths[args.ds_names.index(dsname)]
    out_dir = args.base_out_dir + os.path.basename(ds_dir)
    pm_list = args.pm_list[dsname]
    if len(pm_list) <= 0:
        log.warning(f'No measurements found for datastream [{dsname}]')
        return
    site_name = dsname[0:3]
    log.info(f'[input-directory] [{ds_dir}]')


    # The list of cdf files to process
    path_strs = getPathStrs(ds_dir)
    log.info(f'[total data files] {len(path_strs)}')

    if len(path_strs) == 0:
        log.info(f'No new files for datastream [{dsname}]')
        return

    current_idxs = getSortedFileIndices(args.start_date, args.num_days, path_strs)
    log.debug(f'[current_idxs] {current_idxs}')
    num_to_process = len(current_idxs)
    log.info(f'[files-to-process] [{num_to_process}]')
    log.info(f'[measurements] [{len(pm_list)}] {pm_list}')
    log.info(f'[pngs-to-generate] [{num_to_process * len(pm_list) * 2}]')
    log.info(f'[current-output-directory] [{out_dir}]\n')

    data_file_paths = np.array(path_strs)[current_idxs]
    # data_file_paths = [file_path for file_path in data_file_paths for r in result_list if
    #                    r in act.io.armfiles.read_netcdf(file_path).data_vars.keys()]
    partial_processPm = partial(processPm, copy.deepcopy(args), site_name, out_dir, dsname)

    # TODO: Use multiprocessing.Value to monitor the progress
    #   with counter.get_lock():
    #     counter.value += 1
    with multiprocessing.Pool(int(args.num_t)) as pool:
        image_paths = pool.starmap(partial_processPm, product(data_file_paths, pm_list))
        image_paths = [ i for i in image_paths if i ]
        if len(image_paths) > 0:
            image_paths.sort()
            image_paths = [ list(i) for j, i in groupby(image_paths,
                                                        lambda a: re.search('([a-zA-Z0-9]+\.[a-z0-9]{2}\.\d{8})', a).group())]
            pool.map(combine, image_paths)
        else:
            log.warning(f'[No plots generated for datastream] [{dsname}]')

def readDatastreamsFromSiteTxt(site):
    site_datastreams_file = site+'.txt'
    if os.path.isfile(site_datastreams_file):
        with open(site_datastreams_file, 'r') as site_datastreams:
            return site_datastreams.readlines()
    else:
        err = f'!![ERROR]!! Failed to read site txt file: {site_datastreams_file}'
        log.error(err)
        sys.stderr.write(err)

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
    log.trace(f'[SELECT_PRIMARY_MEASUREMENTS] {SELECT_PRIMARY_MEASUREMENTS} {(tuple(ds_names),)}')

    conn = getDBConnection()
    try:
        with conn.cursor() as curs:
            curs.execute(SELECT_PRIMARY_MEASUREMENTS, (tuple(ds_names),))
            results = curs.fetchall()
        for pm in results:
            ds_dict[pm[0]].append(pm[1])
            # log.info(f'pm: {pm}')
    finally:
        conn.close()
        # print('[CLOSED]')
    return ds_dict

def getEndDates(data_file_paths):
    end_dates = {}
    for path in data_file_paths:
        modified_time = os.path.getmtime(path)
        end_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
        # print(f'[last_modified]::[{path}]', end_date)
        end_dates[os.path.basename(path)] = end_date
    return end_dates

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
    return parser.parse_args()


def main(args):
    sites = args.site_list.split(',')
    log.info('[sites]', sites)

    if args.use_txt_file:
        log.info('-- reading datastreams from site txt --')
        for site in sites:
            args.ds_names = [ ds.strip() for ds in readDatastreamsFromSiteTxt(site) ]
            # print(f'[args.ds_names] {args.ds_names}')
            if args.ds_names is None: continue

            args.data_file_paths = [os.path.join('/data/archive/', site, ds.strip()) for ds in args.ds_names]
            args.data_file_paths = list(filter(lambda p: offsetDays(p, args.num_days), args.data_file_paths))
            # reduce ds_names to only those within data_file_paths
            args.ds_names = [ ds for ds in args.ds_names if any(ds in path for path in args.data_file_paths) ]
            if len(args.ds_names) == 0:
                log.info(f'No recent datastream files in the day range [{args.num_days}] for site [{site}]')
                continue
            log.info(f'[datastreams within {args.num_days} day(s)] {args.ds_names}')
            log.info(f'[total] {len(args.ds_names)}')

            args.start_date = 'current'
            args.end_dates = getEndDates(args.data_file_paths)
            args.pm_list = buildPrimaryMeasurementDict(args.ds_names)

            for ds in args.ds_names:
                getPrimaryForDs(args, ds)

    # TODO: Move this to `if not args.use_txt_file`
    else:
        print('[WARNING] This will attempt to get all datastreams for a given site. This is not recommended and not guaranteed to work. '
              'Please use the --use-txt-file flag and provide a file in the same directory containing a list of datastreams to process, '
              'named like sgp.txt or anx.txt. If you wish to try this anyway, uncomment the proceeding lines in main().')
        sys.exit(1)

    print('This should be one of the last thing printed.')

if __name__ == '__main__':
    started = datetime.now()
    log.info('[BEGIN]', started,'\n--------------------------------------------\n')
    args = getArgs()
    main(args)
    log.info('Done with all!\n')
    elapsed_time = datetime.now() - started
    log.info(f'[Processing time] {elapsed_time}\n\n')
