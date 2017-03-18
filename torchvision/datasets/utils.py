def check_integrity(fpath, md5):
    import hashlib
    if not os.path.isfile(fpath):
        return False
    md5c = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
    if md5c != md5:
        return False
    return True


def download(url, root, filename, md5=None):
    from six.moves import urllib

    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)
