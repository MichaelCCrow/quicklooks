#!/usr/bin/env python
import os
import sys
import argparse
import act
import re
import copy
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from time import process_time
from socket import gethostname
from datetime import datetime, timedelta
from PIL import Image
from loguru import logger as log
from itertools import product, groupby
from functools import partial, partialmethod, wraps
from pathlib import Path

from quicklooks.args import getparser
from settings import DB_SETTINGS

progress_counter = None
total_counter = None

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

def timer(f):
    @wraps(f)
    def wrapper_timer(*args, **kwargs):
        start_time = process_time()
        value = f(*args, **kwargs)
        run_time = process_time() - start_time
        if run_time > 2:
            log.info(f'[time][{f.__name__!r}] {run_time}s')
        return value
    return wrapper_timer


def get_file_date(filePath):
    strDate = os.path.basename(os.path.normpath(filePath)).split('.')[2]
    return datetime.strptime(strDate, '%Y%m%d')

def getDateStr(dateObj):
    return datetime.strftime(dateObj, '%Y%m%d')

def getStartDate(dates, selectedDate):
    return min(dates, key=lambda currDate: abs(currDate - selectedDate))

def offsetDays(file, days=1):
    file = Path(file)
    offset = datetime.fromtimestamp(file.stat().st_mtime) + timedelta(days=int(days))
    return offset > datetime.today()


@timer
def getPathStrs(dataDir):
    return [ os.path.join(dataDir, f) for f in set(os.listdir(dataDir)) if f.endswith('.nc') or f.endswith('.cdf') ]
    # pathlistNC = Path(dataDir).glob('**/*.nc')
    # pathlistNC = list(filter(lambda p: offsetDays(p, days), pathlistNC))
    # pathlistCDF = Path(dataDir).glob('**/*.cdf')
    # pathlistCDF = list(filter(lambda p: offsetDays(p, days), pathlistCDF))

    # pathStrs = list(map(str, pathlistNC))
    # pathStrs += list(map(str, pathlistCDF))

    # return pathStrs


@timer
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
        startIdx = -1 if dateOffset == 0 else -1 * dateOffset
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


def getOutputFilePath(siteName, dataStreamName, baseOutDir, figSizeDir, pmResult, dataFilePath):
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

# TODO: Move this to just above combine()
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


def getOutputUrl(siteName, dataStreamName, baseOutDir, figSizeDir, pmResult, dataFilePath):
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


def getEndDates(data_file_paths):
    end_dates = {}
    for path in data_file_paths:
        modified_time = os.path.getmtime(path)
        end_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
        # print(f'[last_modified]::[{path}]', end_date)
        end_dates[os.path.basename(path)] = end_date
    return end_dates


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


def update_ql_tables(urlStr, dsname, pm, end_dates):
    log.trace(f'[URL STRING] {urlStr}')
    pre_selected_qls_info_insert_query = createPreSelectInsert(dsname, pm, urlStr, endDate=end_dates[dsname])
    giri_inventory_insert_query = createGiriInsert(dsname, pm, endDate=end_dates[dsname])
    conn = getDBConnection()
    if conn is None:
        log.critical(f'Database connection failed. Quicklooks tables were not updated. {urlStr}')
        return
    with conn:
        insert_qls_info(pre_selected_qls_info_insert_query, conn, 'pre_selected_qls_info')
        insert_qls_info(giri_inventory_insert_query, conn, 'giri_inventory')
    conn.close()


fig_sizes = [(1.0, 1.0), (7.4, 4.0)]
fig_size_dirs = ['/.icons', '']


def processPm(args, dsname, data_file_path, pm):
    global progress_counter
    idx_started = datetime.now()
    log.debug(f'[{data_file_path.split("/")[5]}] [{pm}] | {progress_counter.value}/{total_counter.value}')

    args.file_path = data_file_path # required for timeseries function
    fsize = os.path.getsize(data_file_path)
    if fsize > args.max_file_size:  # exclude if file size is > 100MB
        log.warning(f'File too large: {data_file_path} : [{fsize}]')
        return

    site_name = dsname[0:3]
    iconpath = getOutputFilePath(site_name, dsname, args.base_out_dir, fig_size_dirs[0], pm, data_file_path)
    full_plot_path = iconpath.replace(fig_size_dirs[0], fig_size_dirs[1])
    outpaths = [ iconpath, full_plot_path ]

    thumb_exists = os.path.exists(outpaths[0])
    img_exists = os.path.exists(outpaths[1])
    if thumb_exists and img_exists:
        with progress_counter.get_lock():
            progress_counter.value += 2
            log.opt(raw=True).info(f'[{dsname}] {progress_counter.value}/{total_counter.value} | {int((progress_counter.value / total_counter.value) * 100)}%\r')
        # Check if we've already inserted a url for this datastream
        urlStr = outpaths[0].replace(args.base_out_dir, 'https://adc.arm.gov/quicklooks/')
        # TODO: Gather these and run the update/insert queries only once per pm at the end of the multiprocessing loop
        update_ql_tables(urlStr, dsname, pm, args.end_dates)
        return

    started_act_read_cdf = datetime.now()
    try:
        try:
            dataset = act.io.armfiles.read_netcdf(data_file_path)
        except Exception as e:
            log.error(f'[FAILED][ACT][could not read netcdf file using ACT library] {data_file_path} [REASON] {e}')
            return
        if pm not in dataset.data_vars.keys():
            log.debug(f'[SKIPPING][measurement-not-in-cdf] [{pm}] [{os.path.basename(data_file_path)}]')
            dataset.close()
            return
        args.dataset = dataset
    except Exception as e:
        log.critical(f'[FAILED][checking for measurements in cdf file] {data_file_path} [REASON] {e}')
        dataset.close()
        return
    elapsed = datetime.now() - started_act_read_cdf
    if elapsed.total_seconds() > 3:
        log.info(f'[time][act_read_cdf] {elapsed}')

    imgpath = ''

    # required for timeseries plotting method
    args.field = pm
    args.yrange = getyrange(dsname, pm)
    # TODO: Fix/update the yranges in the database
    if args.yrange is None:
        log.warning(f'[Y-Range not found] {dsname} <-> {pm}')

    # TODO: Figure out how to remove the necessity for a loop
    for idx in range(2):
        # fig_size_dir = fig_size_dirs[idx]
        args.figsize = fig_sizes[idx]
        args.out_path = outpaths[idx]
        # args.out_path = getOutputFilePath(site_name, dsname, args.base_out_dir, fig_size_dir,
        #                                   pm, data_file_path)
        with progress_counter.get_lock():
            progress_counter.value += 1

        # TODO: Once Elastic is implemented, remove the table inserts, as they will no longer be needed.
        # TODO: When generating NEW plots, this will not put the URLs in the database since they don't exist. #FIXME
        # if os.path.exists(args.out_path):
        if (idx==0 and thumb_exists) or (idx==1 and img_exists):
            log.opt(raw=True).info(f'[{dsname}] {progress_counter.value}/{total_counter.value} | {int((progress_counter.value/total_counter.value)*100)}%\r')
            urlStr = args.out_path.replace(args.base_out_dir, 'https://adc.arm.gov/quicklooks/')
            update_ql_tables(urlStr, dsname, pm, args.end_dates)
            continue

        if idx == 1:
            args.title = getSegmentName(data_file_path) + " " + pm
            args.show_axis = 'on'
        else:
            args.title = ""
            args.show_axis = 'off'
        action_started = datetime.now()
        try:
            args.action(args) # now executes any methods flagged from command line args
            if idx == 1:
                imgpath = args.out_path
            elif args.index:
                # thumbnail for the .icons sub directory
                thumbnail = args.out_path
                if os.path.getsize(thumbnail) <= 409:
                    log.info(f'[blank] {thumbnail}')
                    log.index(thumbnail+'.blank')
                else:
                    log.index(thumbnail)
            action_time = datetime.now() - action_started
            if action_time.total_seconds() > 30:
                log.info(f'[plot-generated] {args.out_path} | [time][>30s] {action_time.total_seconds()}s')
            else:
                log.info(f'[plot-generated] {args.out_path}')
        except Exception as e:
            # from traceback import print_exc
            # print_exc()
            errmsg = f'[FAILED] [{data_file_path}] [{pm}] -- [REASON] [{e}]'
            log.error(errmsg)
            if os.access('/tmp/bad_datastreams.txt', os.W_OK):
                with open('/tmp/bad_datastreams.txt', 'a+') as file_out:
                    print(args.out_path, file=file_out)
            else:
                with open('logs/bad_datastreams.txt', 'a+') as f:
                    print(errmsg, file=f)
        finally:
            action_time = datetime.now() - action_started
            # if action_time.total_seconds() > 60:
            #     log.info(f'[time][plot-generation][{os.path.basename(args.out_path)}] {action_time.total_seconds()}s')
            log.trace(f'[time][plot-generation] {action_time}')

        # except Exception as e:
        #     log.error(f'[FAILED to PROCESS] {data_file_path} [REASON] {str(e)}')
        #     if os.access('/tmp/bad_datastreams.txt', os.W_OK):
        #         with open('/tmp/bad_datastreams.txt', 'a+') as file_out:
        #             print(args.out_path, file=file_out)
        #     plt.close()
        #     return
    plt.close()

    idx_time = datetime.now() - idx_started
    log.trace(f'[time][process-pm][{pm}] {idx_time}')
    log.complete()

    return imgpath


def getPrimaryForDs(args, dsname, ds_dir, pm_list):
    global total_counter
    log.info('\n**************************************')
    log.info(f'[PROCESSING][datastream] [{dsname}]')
    log.info(f'[input-directory] [{ds_dir}] listing files...')

    # Unfiltered list of cdf files in the datastream directory
    path_strs = getPathStrs(ds_dir)
    if len(path_strs) == 0:
        log.info(f'No new files for datastream [{dsname}]')
        return
    args.dsname = dsname # required for other methods to find the dsname

    current_idxs = getSortedFileIndices(args.start_date, args.num_days, path_strs)
    log.debug(f'[current_idxs] {current_idxs}')
    num_to_process = len(current_idxs)
    log.info(f'[files-to-process] [{num_to_process} out of {len(path_strs)}]')
    log.info(f'[measurements] [{len(pm_list)}] {pm_list}')
    num_pngs_to_generate = num_to_process * len(pm_list) * 2
    log.info(f'[pngs-to-generate] [{num_pngs_to_generate}]')
    # log.info(f'[current-output-directory] [{out_dir}]\n')
    with total_counter.get_lock():
        total_counter.value = num_pngs_to_generate

    data_file_paths = np.array(path_strs)[current_idxs] # [ /data/archive/nsa/nsa30ecorE10.b1/nsa30ecorE10.b1.20210925.000000.cdf, ...]

    # TODO: Consider creating a map/dict/tuple of input to output paths and filtering existing output paths from the list to help improve performance
    partial_processPm = partial(processPm, copy.deepcopy(args), dsname)

    with multiprocessing.Pool(args.num_threads) as pool:
        img_started = datetime.now()
        image_paths = pool.starmap(partial_processPm, product(data_file_paths, pm_list))

        elapsed = datetime.now() - img_started
        log.info(f'[time][process-imgs][{dsname}] {elapsed}')

        image_paths = [ i for i in image_paths if i ]
        if len(image_paths) > 0:
            image_paths.sort()
            image_paths = [ list(i) for j, i in groupby(image_paths,
                                                        lambda a: re.search('([a-zA-Z0-9]+\.[a-z0-9]{2}\.\d{8})', a).group())]
            pool.map(combine, image_paths)
        else:
            log.warning(f'[No plots generated for datastream] [{dsname}]')


def main(args):
    global total_counter
    global progress_counter

    if not args.datastreams: # This should never occur since args are required and mutually exclusive
        print(
            '[WARNING] This will attempt to get all datastreams for a given site. This is not recommended and not guaranteed to work. '
            'Please use the --use-txt-file flag and provide a file in the same directory containing a list of datastreams to process, '
            'named like sgp.txt or anx.txt. If you wish to try this anyway, uncomment the proceeding lines in main().')
        sys.exit(1)
    if not args.index:
        log.warning('Index flag is not set. New plots will not be added to the index.txts and ElasticSearch will be out of sync.')
        sys.stderr.write('Index flag is not set. New plots will not be added to the index.txts and ElasticSearch will be out of sync.\n')

    log.info(f'Running for Datastreams: {args.datastreams}')
    # sites = set([ ds[:3] for ds in args.datastreams ])
    site_ds_grouped = [(site, set(dsnames)) for site, dsnames in groupby(args.datastreams, lambda ds: ds[:3])]
    for site, ds_names in site_ds_grouped:
        log.info('********************************************************\n')
        log.info(f'[Processing][site] [{site}]')
        if ds_names is None: continue
        data_file_paths = [os.path.join('/data/archive/', site, ds.strip()) for ds in ds_names]
        data_file_paths = list(filter(lambda p: offsetDays(p, args.num_days), data_file_paths))

        # reduce ds_names to only those within data_file_paths
        ds_names = [ ds for ds in ds_names if any(ds in path for path in data_file_paths) ]
        if len(ds_names) == 0:
            log.info(f'No recent datastream files in the day range [{args.num_days}] for site [{site}]')
            continue
        log.info(f'[datastreams within {args.num_days} day(s)] {ds_names}')
        log.info(f'[total] {len(ds_names)}')

        args.start_date = 'current'
        args.end_dates = getEndDates(data_file_paths)
        pm_dict = buildPrimaryMeasurementDict(ds_names)
        # args.pm_list = buildPrimaryMeasurementDict(ds_names) # only needed if to use the `mtimeseries`, otherwise, don't assign it to `args`

        num_ds = len(ds_names)
        count = 0

        def checkprogress(count):
            count += 1
            perc = int((count / num_ds) * 100)
            log.info(f'[PROGRESS] {count}/{num_ds} | {perc}% [{site}] completed')
            return count

        for ds in ds_names:
            pm_list = pm_dict[ds]
            if len(pm_list) <= 0:
                log.warning(f'No measurements found for datastream [{ds}]')
                count = checkprogress(count)
                continue
            ds_dir = data_file_paths[ds_names.index(ds)]
            getPrimaryForDs(args, ds, ds_dir, pm_list)
            count = checkprogress(count)
            with total_counter.get_lock():
                total_counter.value = 0
            with progress_counter.get_lock():
                progress_counter.value = 0
    # print('This should be one of the last thing printed.')


class ReadDatastreamTxtsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == '--use-txt-dir' or option_string == '-txtdir':
            setattr(namespace, self.dest, self.datastream_txts())
            return
        txtfiles = [ v for v in values if os.path.isfile(v) ]
        notfiles = [ v for v in values if v not in txtfiles ]
        if notfiles: log.warning(f'[not files] {notfiles}')
        ds = self.read_txts(txtfiles)
        setattr(namespace, self.dest, ds)

    def read_txts(self, txts):
        ds = []
        for txt in txts:
            with open(txt, 'r') as f:
                lines = f.read().splitlines()
                ds += lines
                # for line in lines: ds.append(line)
        log.debug(ds)
        return ds

    def datastream_txts(self, txtdir='txt'):
        if not os.path.isdir(txtdir):
            log.error(f'[directory does not exist] {txtdir}')
            raise argparse.ArgumentError(None, message=f'["{txtdir}/" directory does not exist] '
                                                       f'The {txtdir} directory is required to exist in the same relative '
                                                       f'directory as this script in order to use this argument.\n'
                                                       f'Run {os.path.basename(sys.argv[0])} -h for usage details.')
        txts = os.listdir(txtdir)
        # txt files must be a 3-letter site code with '.txt' extension
        pat = re.compile('^[a-z]{3}\.txt$')
        # nomatch = [ t for t in txts if not re.match(pat, t) ]
        txts = [os.path.join(txtdir, t) for t in txts if re.match(pat, t)]
        txts.sort()
        log.debug(f'[using txts] {txts}')
        return self.read_txts(txts)


def getArgs():
    parent = getparser()
    subparser = argparse.ArgumentParser(description='Create GeoDisplay Plot', parents=[parent])
    parser = subparser.add_argument_group('wrapper script arguments')
    parser.add_argument('-days', '--num-days', type=int,
                        help='Number of days offset from latest file date to process')
    parser.add_argument('-nt', '--num-threads', type=int,
                        help='Max number of threads')
    parser.add_argument('-maxFs', '--max-file-size', type=int, default=100000000, dest='max_file_size',
                        help='Max file size in number of bytes - default is 100000000 (100MB)')
    parser.add_argument('--log-file', type=str,
                        default=sys.stderr if gethostname()=='mcmbpro' else 'logs/act.log',
                        help='File to write output logs to. Should end with ".log". (default: %(default)s)')
    parser.add_argument('-baseOut', '--base-out-dir', type=str, default='/data/ql_plots/',
                        help='Base Out Directory to use for saving Plot. Do not use default. (default %(default)s)')
    parser.add_argument('--debug-log-file', type=str, default='logs/debug.log',
                        help='Full file path to debug log file. (default: %(default)s)')
    parser.add_argument('--index', '--write-index-txt', dest='index', action='store_true',
                        help='Flag to indicate that index files should be written to for ElasticSearch to pick up. '
                             'The base directory is the same as the value for the --base-out-dir argument. '
                             'NOTE: This argument is primarily intended to be omitted for debugging and testing and '
                             'should ALWAYS be true in production, or files will be missed.')
    dsargs = subparser.add_argument_group('datastream selection required arguments').add_mutually_exclusive_group(required=True)
    # TODO: Add functionality for this
    # dsargs.add_argument('-sites2', '--site-list', nargs='+', dest='sites',
    #                     help='Sites for which to process datastreams, excluding the following data levels: .a1 .a0 .00 | Example: -sites sgp nsa ena')
    dsargs.add_argument('-D', '--datastreams', nargs='+', metavar='datastream',
                        help='Datastreams to process. Provide each datastream separated by a space. Example: -D sgp30ebbrE10.b1 nsa30ebbrE10.b1')
    dsargs.add_argument('-txtfiles', '--datastream-txts', nargs='+', dest='datastreams', metavar='file.txt', action=ReadDatastreamTxtsAction,
                        help='Provide a space separated list of paths to datastream txt files. Example: -txtfiles nsa.txt subdir/sgp.txt tempENA.txt')
    dsargs.add_argument('-txtdir', '--use-txt-dir', nargs=0, dest='datastreams', action=ReadDatastreamTxtsAction,
                        help='Signals to to the script to use only the txt files found in the relative subdirectory "txt".'
                             'Only files named with 3-letter site code and ".txt" extension will be used. Others will be ignored.')

    '''These are not used from the command line - they are set later in the program'''
    # parser.add_argument('-f', '--file-path', type=str, help='File to use for creating Plot')
    # parser.add_argument('-o', '--out_path', type=str, help='File path to use for saving image')
    # parser.add_argument('-fd', '--field', type=str, default=None, help='Name of the field to plot')
    return subparser.parse_args()


# TODO: Add proper README.md
if __name__ == '__main__':
    started = datetime.now()
    if os.getcwd().startswith('/apps/adc/act/quicklooks/dailyquicklooks'): log.remove() # remove log from arg parsing if prod
    args = getArgs()

    log.remove()
    log.add(args.log_file, level='INFO', enqueue=True, colorize=True, rotation='100 MB', compression='zip',
            format='<g>{time:YYYY-MM-DD HH:mm:ss!UTC}</g> | <lvl>{level: >4}</lvl> | <lvl>{message}</lvl>')
    log.info('[BEGIN]', started, '\n--------------------------------------------\n')
    log.add(args.debug_log_file, enqueue=True, colorize=True, rotation='100 MB', compression='zip',
            filter=lambda record: record['level'].name == 'DEBUG')

    if args.index:
        def _get_index_file(index_base_dir, plot_file_path):
            year = plot_file_path.split('/')[4] # if plot_file_path.startswith('/var/ftp/quicklooks') else re.search('(?<=\/)\d{4}(?=\/)', plot_file_path).group()
            index_file_path = os.path.join(index_base_dir, year, 'index.txt')
            print(plot_file_path, file=open(index_file_path, 'a'), end='')  # 3
        def _get_dev_index_file(index_base_dir, plot_file_path): # Only used for index file testing
            year = os.path.basename(plot_file_path).split('.')[2][:4]
            index_file_path = os.path.join(index_base_dir, year, 'index.txt')
            print(plot_file_path, file=open(index_file_path, 'a'), end='')

        log.level('INDEX', no=2)
        log.__class__.index = partialmethod(log.__class__.log, 'INDEX')
        partial__get_index_file = partial(_get_index_file, args.base_out_dir) \
            if args.base_out_dir.startswith('/var/ftp/quicklooks') else partial(_get_dev_index_file, args.base_out_dir)
        log.add(partial__get_index_file, enqueue=True, colorize=False,
                filter=lambda record: record['level'].name == 'INDEX',
                level='INDEX',
                format='{message}')

    progress_counter = multiprocessing.Value('i', 0)
    total_counter = multiprocessing.Value('i', 0)

    main(args)

    log.info(f'[Processing time] {datetime.now() - started}\n--------------------------------------------------------------\n')
