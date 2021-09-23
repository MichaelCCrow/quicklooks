import os
from os.path import join, basename, dirname, exists, isdir
import tarfile
import argparse
import psycopg2
from datetime import datetime
from multiprocessing import Pool, Value
from PIL import Image
from loguru import logger as log
from settings import DB_SETTINGS

logc = log.opt(colors=True)
''' Default output directory - overridden by args.output_dir '''
ftp_root = '/var/ftp'
''' Default src directory '''
archive_root = '/data/archive'
url_root = 'https://adc.arm.gov'
total_datastreams = 0
num_tars_in_datastream = 0
totalcount = 0
counter = None
tarcounter = None
completed = []


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
    if not isdir(dirname(dest)):
        os.makedirs(dirname(dest))
    image.save(dest, optimize=True)


def rename(out_path, img):
    '''Tarred images are poorly named - this method removes the garbage from the image file name and renames it in place.'''
    parts = img.split('.')
    modified_name = '.'.join(parts[:4] + parts[-1:])
    # log.trace(modified_name)
    curpath = join(out_path, img)
    outpath = join(out_path, modified_name)
    # modified_name = '.'.join(img.split('.')[:4]) + os.path.splitext(img)[1]
    # i = img.replace('.', '+', 3).find('.')
    log.log('SUPERTRACE', f'[from]{curpath} -> [to]{outpath}')
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
    global total_datastreams
    # if True: return [('nsatwrcam40mC1.a1', 'jpg')]
    q = "SELECT datastream, var_name FROM arm_int2.datastream_var_name_info WHERE var_name IN ('jpg', 'png')"
    # TODO: Upgrade this so that only new FILES are processed.
    #   As of now, all this does is select the datastreams, so every single file for all time will still be processed, but only for the recent datastreams returned by the query.
    if days != -1:
        log.debug(f'[limiting results to {days}]')
        q += f''' AND end_date > NOW() - INTERVAL '{days} DAYS' '''
    else: log.warning('[running for all time]')

    conn = getDBConnection()
    with conn.cursor() as cur:
        cur.execute(q)
        total_datastreams = cur.rowcount
        log.info(f'[datastreams found] {total_datastreams}')
        results = cur.fetchall() # [('datastream.b1', 'jpg'), ... ]
        log.debug(f'[ALL image datastreams from database] {results}')
        cur.close()
    conn.close()
    return results


def get_output_dir(srcfile, datastream):
    ''' Build output directory by getting the date portion of the file name, and using the pieces of the datastream name to construct the ftp output path. '''
    dsdate = basename(srcfile).split('.')[2]
    out_path = join(ftp_root, 'quicklooks', dsdate[:4], datastream[:3], datastream, f'{datastream}.{dsdate[:6]}')
    if not isdir(out_path):
        os.makedirs(out_path)
    return out_path


"""
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
"""

'''
def gettotal(tfile):
    with tarfile.open(tfile, 'r') as tar:
        return len(tar.getnames())
'''


def update_thumbnail(datastream, thumb):
    log.info(f'[UPDATING THUMBNAIL] {thumb}')
    update_pre_selected(datastream, thumb)
    update_giri_inventory(datastream)
    completed.append((datastream, thumb))


def create_symlinks_for_existing_images(imgfiles, datastream, img_type):
    log.info('[IMAGE FILES FOUND - SYMLINKING...]')
    started_creating_symlinks = datetime.now()
    imgfiles.sort()
    imgfiles.reverse()
    log.trace(f'{imgfiles[:5]}')
    thumb = ''
    i = 0
    ''' Create symlinks to the images in the ftp area '''
    for imgfile_path in imgfiles[:10]:
        dest = join(get_output_dir(imgfile_path, datastream), basename(imgfile_path))
        log.debug(f'[src] {imgfile_path} --> [dest] {dest}')
        if exists(dest):
            log.warning(f'[already exists] {dest}')
            continue
        os.symlink(imgfile_path, dest)
        thumb = join(dirname(dest), '.icons', basename(dest))
        log.debug(f'[CREATE ICON THUMB] {dest} -> {thumb}')
        create_thumbnail(bigimg=dest, dest=thumb)
        if i == 0:
            i = 1
            logc.info(f'[<g>Thumbnail should exist here</>] {thumb}')
            if not exists(thumb):
                log.error(f'[THUMBNAIL NOT FOUND] {thumb} - Before attempting to update the qls thumbnail, it could not be found.')
                continue
            thumb = thumb.replace(ftp_root, url_root)
            log.info(f'[UPDATING {img_type} THUMBNAIL] {thumb}')
    update_thumbnail(datastream, thumb)
    logc.info(f'[<m>time</>][create-symlinks] {datetime.now() - started_creating_symlinks}')


def extract_tar(tfile, out_path):
    global counter
    global tarcounter
    urls = []
    with tarfile.open(tfile, 'r') as tar:
        img_names = tar.getnames()
        log.trace(img_names)

        ''' Keep count of how many are extracted '''
        total_images = len(img_names)
        logc.info(f'[image-count][{tfile}] <y>{total_images}</>')
        with counter.get_lock():
            counter.value += total_images

        tar.extractall(out_path)

        ''' Images have been extracted into quicklooks ftp '''
        for img in img_names:
            # urls.append(build_thumbs(out_path, img))
            ''' Looping through each image name, rename them appropriately in place in the ftp '''
            modified_name = rename(out_path, img)
            ''' Use the base output path to append .icons for the thumbnail output path '''
            thumb = join(out_path, '.icons', basename(modified_name))
            log.trace(f'[CREATING icon in .icons dir] {thumb}')
            create_thumbnail(bigimg=modified_name, dest=thumb)
            ''' Substitute /var/ftp for https link '''
            url = thumb.replace(ftp_root, url_root)
            urls.append(url)
    with tarcounter.get_lock():
        tarcounter.value += 1
        logc.info(f'[<c>tars-extracted</>] {tarcounter.value}/{num_tars_in_datastream} | <g>{int((tarcounter.value/num_tars_in_datastream) * 100)}%</>') # <d>[{tfile.split('/')[4]}] completed</>''')
    return urls


def process_tars(tarfiles, datastream):
    global counter
    global totalcount
    global num_tars_in_datastream
    num_tars_in_datastream = len(tarfiles)
    started_processing_tar_files = datetime.now()
    tarfiles.sort()
    tarfiles.reverse()

    ''' A tuple of the source tar file path, and the destination directory - used for parallel iteration '''
    in_out_paths = [(tarfile_path, get_output_dir(tarfile_path, datastream)) for tarfile_path in
                    [f for f in tarfiles]]
    log.trace(f'[in_out_paths] {in_out_paths[:5]}')

    ''' For each tar file, extract images and collect the url for the thumbnails '''
    with Pool(args.processes) as pool:
        log.info('[processing tar files]')
        # totalimages = sum(pool.map(gettotal, tarfiles))
        # log.warning(f'[TOTAL IMAGES IN TAR FILES] {totalimages}')
        thumbs = pool.starmap(extract_tar, in_out_paths)
        thumbs = [t for t in thumbs if t]
        logc.info(f'[<g>THUMBS PROCESSED</>] {len(sum(thumbs, []))}')
    logc.info(f'[<g>TARRED IMAGES EXTRACTED</>] {counter.value}')
    with counter.get_lock():
        totalcount += counter.value
        counter.value = 0
    update_thumbnail(datastream, thumbs[0][0])
    logc.info(f'[<m>time</>][process-tarfiles] {datetime.now() - started_processing_tar_files}')


def main(args):
    global tarcounter
    datastreams = args.datastreams or get_img_datastreams(args.days)[:2]
    # datastreams = [('marcamseastateM1.a1', 'jpg')] # used to test symlinks
    log.debug(datastreams)
    prog = 0

    for datastream, img_type in datastreams:
        with tarcounter.get_lock():
            tarcounter.value = 0
        prog += 1
        logc.info(f'[<c>datastreams-processed</>] [{prog}/{total_datastreams}]')

        datastream_dir = join(archive_root, datastream[:3], datastream) # /data/archive/{site}/{datastream}
        if not isdir(datastream_dir):
            log.warning(f'[DOES NOT EXIST] {datastream_dir}')
            continue

        files = os.listdir(datastream_dir)
        log.info(f'[{datastream_dir}] {len(files)}')

        tarfiles = [ join(datastream_dir, file) for file in files if file.endswith('.tar') ]
        if len(tarfiles) > 0:
            process_tars(tarfiles, datastream)

        imgfiles = [ join(datastream_dir, file) for file in files if file.endswith(img_type) ]
        if len(imgfiles) > 0:
            create_symlinks_for_existing_images(imgfiles, datastream, img_type)

    log.info('[DONE]')
    logc.info(f'[<g>TOTAL TARRED IMAGES EXTRACTED</>] {totalcount}')
    log.info(f'[datastreams processed] {len(completed)}\n[datastreams completed]')
    for c in completed: log.info(c)
    log.info('**==========================================**')


if __name__ == '__main__':
    counter = Value('i', 0)
    tarcounter = Value('i', 0)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
         description='''Tarred Image Extractor
    This script extracts jpg and png images from tar files in /data/archive to the quicklooks ftp. 
    Any images found that have already been extracted have symlinks created instead.''',
         epilog='''Example usages:
    Supply datastreams as values to the -D [--datastreams] argument directly:\n
        $ %(prog)s -D sgp30ebbrE10.b1 nsatwrcam40mC1.a1\n
    Use the -A [--all] argument to process all datastreams with image var names:\n
        $ %(prog)s -A''')
    parser.add_argument('-n', '--days', default=-1, type=int,
                        help='Specify the number of days back to check for new files. The default, -1, indicates that files for all time should be processed. (default: %(default)s)')
    parser.add_argument('-o', '--output-dir', default=ftp_root, type=str,
                        help='The root output directory. This argument is primarily intended for testing and debugging. (default: %(default)s)')
    parser.add_argument('-p', '--processes', default=8, type=int,
                        help='Number of parallel processes to spin up while extracting tars. (default: %(default)s)')

    dsargs = parser.add_argument_group('datastream selection required arguments').add_mutually_exclusive_group(required=True)
    dsargs.add_argument('-D', '--datastreams', nargs='+', metavar='datastream',
                        help='Datastreams to process. Provide each datastream separated by a space. Example: -D sgp30ebbrE10.b1 nsa30ebbrE10.b1')
    dsargs.add_argument('-A', '--all', action='store_true',
                        help='Flag to indicate that all datastreams found in the database with variables "jpg" and "png" are to be processed.')

    logargs = parser.add_argument_group('logging options').add_mutually_exclusive_group()
    logargs.add_argument('-i', '--info',  dest='loglevel', const='INFO',  action='store_const', help='Set logging output to INFO')
    logargs.add_argument('-d', '--debug', dest='loglevel', const='DEBUG', action='store_const', help='Set logging output to DEBUG')
    logargs.add_argument('-t', '--trace', dest='loglevel', const='TRACE', action='store_const', help='Set logging output to TRACE')
    logargs.add_argument('-s', '--supertrace', dest='loglevel', const='SUPERTRACE', action='store_const',
                         help='Set logging output to SUPERTRACE. Do not use this unless you want to see millions of file paths logged.')
    # logargs.add_argument('-l', '--log', dest='loglevel', choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    ftp_root = args.output_dir

    log.remove()
    log.add('logs/img_extractor.log', level=args.loglevel or 'DEBUG', enqueue=True, colorize=True, rotation='100 MB', compression='zip',
            format='<g>{time:YYYY-MM-DD HH:mm:ss!UTC}</g> | <lvl>{level: >4}</lvl> | <lvl>{message}</lvl>')
    log.level('SUPERTRACE', no=1, color='<dim>')

    main(args)
