
if qt.config['setup_config_dir'] is not None:
    setup_config = qt.config['setup_config_dir']
    print('Executing user config file: %s' %setup_config)
    exec(compile(open(setup_config).read(), setup_config, 'exec'))

if qt.config['startdir'] is not None:
    os.chdir(qt.config['startdir'])

if qt.config['startscript'] is not None:
    _scripts = qt.config['startscript']
    if type(_scripts) not in (list, tuple):
        _scripts = [_scripts, ]
    for _s in _scripts:
        if os.path.isfile(_s):
            print('Executing (user startscript): %s' % _s)
            exec(compile(open(_s).read(), _s, 'exec'))
        else:
            logging.warning('Did not find startscript "%s", skipping', _s)

if qt.config['exitscript'] is not None:
    _scripts = qt.config['exitscript']
    if type(_scripts) not in (list, tuple):
        _scripts = [_scripts, ]
    for _s in _scripts:
        if os.path.isfile(_s):
            qt.flow.register_exit_script(_s)
        else:
            logging.warning('Did not find exitscript "%s", will not be executed', _s)

# Add script directories. Read index and put in namespace
if qt.config['scriptdirs'] is not None:
    for dirname in qt.config['scriptdirs']:
        qt.scripts.add_directory(dirname)
    qt.scripts.scripts_to_namespace(globals())

# Start IPython command logging if requested
if qt.config['ipython_logfile'] not in (None, ''):
    _ip = get_ipython()
    _ip.IP.logger.logstart(logfname=qt.config['ipython_logfile'], logmode='append')