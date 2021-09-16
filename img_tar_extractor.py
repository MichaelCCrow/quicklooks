import os
import tarfile
# import argparse
import psycopg2
from PIL import Image
from loguru import logger as log
from settings import DB_SETTINGS

ftp_root = '/var/ftp'
url_root = 'https://adc.arm.gov'
total = 0

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
    curpath = os.path.join(out_path, img)
    out_file = os.path.join(out_path, modified_name)
    # modified_name = '.'.join(img.split('.')[:4]) + os.path.splitext(img)[1]
    # i = img.replace('.', '+', 3).find('.')
    # log.info(out_file)
    log.trace(f'[from]{curpath} -> [to]{out_file}')
    os.rename(curpath, out_file)
    return out_file

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
    log.trace(q)
    conn = getDBConnection()
    with conn.cursor() as cur:
        cur.execute(q)
        cur.close()
    conn.close()

def extractor(datastream, typ):
    log.trace(f'[ds] {datastream} : [type] {typ}')
    archive_path = os.path.join('/data/archive', datastream[:3], datastream)
    if not os.path.isdir(archive_path):
        log.warning(f'Path {archive_path} does not exist')
        return
    # files = [ os.listdir(archive_path)[0] ] # use this to test a single tar file
    files = os.listdir(archive_path)
    files = [ f for f in files if f.endswith('.tar') ]
    # TODO: Handle the conditions where there are no tar files, but the images themselves.
    files.sort()
    files.reverse() # get the newest ones first
    log.trace(files)
    urls = []

    for file in files:
        log.info(file)
        dsdate = file.split('.')[2]
        out_path = os.path.join('/var/ftp/quicklooks', dsdate[:4], datastream[:3], datastream,
                                f'{datastream}.{dsdate[:6]}')
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        tfile = os.path.join(archive_path, file)
        if not tarfile.is_tarfile(tfile):
            print(f'[ERROR][not-a-tar-file] {file}')
            continue

        with tarfile.open(tfile, 'r') as tar:
            img_names = tar.getnames()
            log.trace(img_names)
            # log.info(os.path.commonprefix(img_names))
            tar.extractall(out_path)

            ''' Images have been extracted into quicklooks ftp '''
            for img in img_names:
                ''' Looping through each image name, rename them appropriately in place in the ftp '''
                modified_name = rename(out_path, img)
                ''' Use the base output path to append .icons for the thumbnail output path '''
                thumb = os.path.join(out_path, '.icons', os.path.basename(modified_name))
                create_thumbnail(bigimg=modified_name, dest=thumb)
                ''' Substituted /var/ftp for https link '''
                url = thumb.replace(ftp_root, url_root)
                urls.append(url)
            # log.debug(urls)
    return urls

def get_img_datastreams():
    global total
    if True: return [('nsatwrcam40mC1.a1', 'jpg')]
    q = '''SELECT datastream, var_name FROM arm_int2.datastream_var_name_info WHERE var_name IN ('jpg', 'png')'''
    conn = getDBConnection()
    with conn.cursor() as cur:
        cur.execute(q)
        total = cur.rowcount
        results = cur.fetchall() # [('datastream.b1', 'jpg'), ... ]
        cur.close()
    conn.close()
    return results

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('dsfile', type=argparse.FileType('r', encoding='utf-8'),
    # parser.add_argument('-d', '--datastreams', type=argparse.FileType('r', encoding='utf-8'),
    #                     help='Path to a line separated file of datatreams to process.')
    # args = parser.parse_args()
    # datastreams = args.datastreams or get_img_datastreams()
    # global total
    # if total == 0:
    #     datastreams = list(datastreams)
    #     total = len(datastreams)

    datastreams = get_img_datastreams()

    log.info(f'[datastreams] {total}')
    log.debug(datastreams)
    prog = 0

    for ds in datastreams:
        prog += 1
        log.info(f'[progress] [{prog}/{total}]')
        thumbs = extractor(*ds) #, start, end)
        thumbs = [ t for t in thumbs if t ]
        # update_pre_selected(ds, thumbs[0])

if __name__ == '__main__':
    main()