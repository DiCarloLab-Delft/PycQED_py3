import os
import sys

def insert_in_file_list(entries, entry, ignore_list):
    adddir, addname = entry
    if os.path.splitext(addname)[1] != ".py":
        return

    for start in ignore_list:
        if addname.startswith(start):
            return

    index = 0
    for (dir, name) in entries:
        if name[0] > addname[0] or (name[0] == addname[0] and name[1] > addname[1]):
            entries.insert(index, entry)
            break
        index += 1

    if index == len(entries):
        entries.append(entry)

def get_shell_files(path, ignore_list):
    ret = []

    entries = os.listdir(path)
    for i in entries:
        if len(i) > 0 and i[0] == '.':
            continue

        if os.path.isdir(i):
            subret = get_shell_files(os.path.join(path, i))
            for j in subret:
                insert_in_file_list(ret, j, ignore_list)
        else:
            insert_in_file_list(ret, (path, i), ignore_list)

    return ret

def show_start_help():
    print('Usage: qtlab <options> <directory> <files>')
    print('\t<directory> is an optional directory to start in')
    print('\t<files> is an optional list of scripts to execute')
    print('\tOptions:')
    print('\t--help\t\tShow this help')
    print('\t-i <pattern>\tIgnore shell scripts starting with <pattern>')
    print('\t\t\t(can be used multiple times')

    import IPython
    ip_version = IPython.__version__.split('.')
    if int(ip_version[0]) > 0 or int(ip_version[1]) > 10:
        ip = IPython.core.ipapi.get()
        ip.exit()
    else:
        ip = IPython.ipapi.get()
        ip.magic('Exit')

def do_start():
    basedir = os.path.split(os.path.dirname(sys.argv[0]))[0]
    sys.path.append(os.path.abspath(os.path.join(basedir, 'source')))

    ignorelist = []
    i = 1

    global __startdir__
    __startdir__ = None
    # FIXME: use of __startdir__ is spread over multiple scripts:
    # 1) source/qtlab_client_shell.py
    # 2) init/02_qtlab_start.py
    # This should be solved differently
    while i < len(sys.argv):
        if os.path.isdir(sys.argv[i]):
            __startdir__ = sys.argv[i]
        elif sys.argv[i] == '-i':
            i += 1
            ignorelist.append(sys.argv[i])
        elif sys.argv[i] == '--help':
            show_start_help()
            return []
        i += 1

    filelist = get_shell_files(os.path.join(basedir, 'init'), ignorelist)
    return filelist

if __name__ == '__main__':
    print('Starting QT Lab environment...')
    filelist = do_start()
    for (dir, name) in filelist:
        filename = '%s/%s' % (dir, name)
        print('Executing %s...' % (filename))
        try:
            exec(compile(open(filename).read(), filename, 'exec'))
        except SystemExit:
            break

    try:
        del filelist, dir, name, filename
    except:
        pass

