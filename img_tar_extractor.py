import os
from os.path import join, basename
import tarfile
import argparse
import psycopg2
from multiprocessing import Pool, Value
from PIL import Image
from loguru import logger as log
from settings import DB_SETTINGS
# except:
#     import sys
#     sys.path.append(os.path.abspath(join('..')))
#     from settings import DB_SETTINGS

ftp_root = '/var/ftp'
archive_root = '/data/archive'
url_root = 'https://adc.arm.gov'
total = 0
# tarred_images = 0
counter = None

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
    except (Exception, psycopg2.DatabaseError) as error:
        log.error(f'[ERROR] Database connection failed. {error}')
        return None
    finally:
        if dbConnection is not None:
            return dbConnection


def create_thumbnail(bigimg, dest):
    image = Image.open(bigimg)
    image.thumbnail(size=(100, 100))
    if not os.path.isdir(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))
    image.save(dest, optimize=True)


def rename(out_path, img):
    parts = img.split('.')
    modified_name = '.'.join(parts[:4] + parts[-1:])
    log.trace(modified_name)
    curpath = join(out_path, img)
    outpath = join(out_path, modified_name)
    # modified_name = '.'.join(img.split('.')[:4]) + os.path.splitext(img)[1]
    # i = img.replace('.', '+', 3).find('.')

    # log.trace(f'[from]{curpath} -> [to]{outpath}')
    os.rename(curpath, outpath)
    return outpath

def query_insert(q):
    log.debug(q)
    conn = getDBConnection()
    with conn.cursor() as cur:
        cur.execute(q)
        cur.close()
    conn.commit()
    conn.close()

def update_giri_inventory(datastream):
    q = f'''
    INSERT INTO color.giri_inventory
          (datastream, var_name, ql_var_name, measurement, start_date, end_date)
    SELECT DISTINCT TRIM('{datastream}'),
        dsinfo.var_name,
        dsinfo.var_name,
        dsinfo.primary_measurement,
        dsinfo.start_date,
        dsinfo.end_date
    FROM arm_int2.datastream_var_name_info dsinfo
    WHERE dsinfo.datastream = '{datastream}'
    ON CONFLICT ON CONSTRAINT giri_inventory_pkey
    DO UPDATE SET end_date = EXCLUDED.end_date
    '''
    query_insert(q)
    log.info('[GIRI INSERT COMPLETE]')

def update_pre_selected(datastream, url):
    q = f'''
    INSERT INTO arm_int2.pre_selected_qls_info
        (datastream, var_name, ql_var_name, start_date, end_date, thumbnail_url)
    SELECT DISTINCT TRIM('{datastream}'),
        dsinfo.var_name,
        dsinfo.var_name,
        dsinfo.start_date,
        dsinfo.end_date,
        '{url}'
    FROM arm_int2.datastream_var_name_info dsinfo
    WHERE dsinfo.datastream = '{datastream}'
    ON CONFLICT ON CONSTRAINT pre_selected_qs_pk
    DO UPDATE SET end_date = EXCLUDED.end_date
    '''
    query_insert(q)


def get_img_datastreams(days=-1):
    global total
    # if True: return [('nsatwrcam40mC1.a1', 'jpg')]
    q = '''SELECT datastream, var_name FROM arm_int2.datastream_var_name_info WHERE var_name IN ('jpg', 'png')'''
    # TODO: Upgrade this so that only new FILES are processed.
    #   As of now, all this does is select the datastreams, so every single file for all time will still be processed, but only for the recent datastreams returned by the query.
    if days != -1:
        log.debug(f'[limiting results to {days}]')
        q += f''' AND end_date > NOW() - INTERVAL '{days} DAYS' '''
    else: log.info("[running for all time]")

    conn = getDBConnection()
    with conn.cursor() as cur:
        cur.execute(q)
        total = cur.rowcount
        results = cur.fetchall() # [('datastream.b1', 'jpg'), ... ]
        log.debug(f'[datastreams with images] {results}')
        cur.close()
    conn.close()
    return results


def get_out_path(srcfile, datastream):
    dsdate = basename(srcfile).split('.')[2]
    out_path = join(ftp_root, 'quicklooks', dsdate[:4], datastream[:3], datastream, f'{datastream}.{dsdate[:6]}')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    return out_path


def build_thumbs(out_path, img):
    ''' Looping through each image name, rename them appropriately in place in the ftp '''
    modified_name = rename(out_path, img)
    ''' Use the base output path to append .icons for the thumbnail output path '''
    thumb = join(out_path, '.icons', basename(modified_name))
    log.trace('[CREATING THUMBNAIL]')
    create_thumbnail(bigimg=modified_name, dest=thumb)
    ''' Substitute /var/ftp for https link '''
    url = thumb.replace(ftp_root, url_root)
    return url


def gettotal(tfile):
    with tarfile.open(tfile, 'r') as tar:
        return len(tar.getnames())


def extract_tar(tfile, out_path):
    global counter
    urls = []
    with tarfile.open(tfile, 'r') as tar:
        img_names = tar.getnames()
        log.trace(img_names)

        ''' Keep count of how many are extracted '''
        total_images = len(img_names)
        log.info(f'[# IMAGES][{tfile}] {total_images}')
        with counter.get_lock():
            counter.value += total_images

        # log.info('[EXCTRACTING]')
        tar.extractall(out_path)

        ''' Images have been extracted into quicklooks ftp '''
        for img in img_names:
            # urls.append(build_thumbs(out_path, img))
            ''' Looping through each image name, rename them appropriately in place in the ftp '''
            modified_name = rename(out_path, img)
            ''' Use the base output path to append .icons for the thumbnail output path '''
            thumb = join(out_path, '.icons', basename(modified_name))
            # log.trace('[CREATING THUMBNAIL]')
            create_thumbnail(bigimg=modified_name, dest=thumb)
            ''' Substitute /var/ftp for https link '''
            url = thumb.replace(ftp_root, url_root)
            urls.append(url)
    return urls


def main(args):
    datastreams = get_img_datastreams(args.days)[:2]
    # datastreams = [('marcamseastateM1.a1', 'jpg')]

    log.info(f'[datastreams] {total}')
    log.debug(datastreams)
    prog = 0

    ''' For each datastream name '''
    for datastream, img_type in datastreams:
        prog += 1
        i = 0
        log.info(f'[progress] [{prog}/{total}]')

        datastream_dir = join(archive_root, datastream[:3], datastream) # /data/archive/{site}/{datastream}
        if not os.path.isdir(datastream_dir):
            log.warning(f'[DOES NOT EXIST] {datastream_dir}')
            continue

        for root, dirs, files in os.walk(datastream_dir):
            log.info(f'[{datastream_dir}] {len(files)}')
            log.info('[building tar file list...]')
            tarfiles = [ join(root, file) for file in files if tarfile.is_tarfile(join(root, file)) ]
            if len(tarfiles) > 0: log.info('[TAR FILES FOUND]')
            tarfiles.sort()
            tarfiles.reverse()

            log.info('[building list of src paths and output paths...]')
            ''' A tuple of the source tar file path, and the destination directory - used for parallel iteration '''
            in_out_paths = [ ( tfile_path, get_out_path(tfile_path, datastream) ) for tfile_path in tarfiles ]

            ''' For each tar file, extract images and collect the url for the thumbnails '''
            with Pool(8) as pool:
                log.info('[processing tar files]')
                # totalimages = sum(pool.map(gettotal, tarfiles))
                # log.warning(f'[TOTAL IMAGES IN TAR FILES] {totalimages}')
                # out_path = get_out_path(tfile_path, datastream)
                # thumbs = extract_tar(tfile_path, out_path)
                thumbs = pool.starmap(extract_tar, in_out_paths)
                thumbs = [ t for t in thumbs if t ]
                log.warning(f'[THUMBS PROCESSED] {len(sum(thumbs, []))}')

            if len(thumbs) > 0 and i == 0:
                i = 1
                log.info(f'[UPDATING THUMBNAIL] {thumbs[0][0]}')
                update_pre_selected(datastream, thumbs[0][0])
                update_giri_inventory(datastream)
            log.info(f'[TOTAL IMAGES TARRED] {counter.value}')

            log.info('[building list of image files...]')
            ''' Get a list of image files '''
            imgfiles = [ join(root, file) for file in files if file.endswith(img_type) ]
            if len(imgfiles) > 0: log.info('[IMAGE FILES FOUND - SYMLINKING]')

            ''' Create symlinks to the images in the ftp area '''
            for imgfile in imgfiles[:10]:
                dest = join(get_out_path(imgfile, datastream), imgfile)
                log.debug(f'[src] {imgfile} --> [dest] {dest}')
                os.symlink(imgfile, dest)
                if i == 0:
                    i = 1
                    thumb = dest.replace(ftp_root, url_root)
                    log.info(f'[UPDATING img THUMBNAIL] {thumb}')
                    update_pre_selected(datastream, thumb)
                    update_giri_inventory(datastream)
    log.info('[DONE]')

        # thumbs = extractor(*ds) #, start, end)
        # thumbs = [ t for t in thumbs if t ]
        # update_pre_selected(ds, thumbs[0])

if __name__ == '__main__':
    log.remove()
    log.add('logs/img_extractor.log', level='DEBUG', enqueue=True, colorize=True, rotation='100 MB', compression='zip',
            format='<g>{time:YYYY-MM-DD HH:mm:ss!UTC}</g> | <lvl>{level: >4}</lvl> | <lvl>{message}</lvl>')
    counter = Value('i', 0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--days', default=-1, type=int, help='Specify the number of days back to check for new files. Default is -1, which indicates that files for all time should be processed.')
    args = parser.parse_args()
    main(args)