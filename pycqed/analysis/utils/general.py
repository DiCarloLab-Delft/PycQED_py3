import subprocess


def get_git_revision_hash():
    import logging
    import subprocess
    try:
        hash =  subprocess.check_output(['git', 'rev-parse','--short=7', 'HEAD'])
    except:
        logging.warning('Failed to get Git revision hash, using 00000 instead')
        hash = '00000'

    return hash


#This is code from Kwant that Anton showed me (Adriaan), it is located 
# at http://git.kwant-project.org/kwant/tree/setup.py.
# it should in the future replace the current git get revision hash function. 

# This is an exact copy of the function from kwant/version.py.  We can't import
# it here (because Kwant is not yet built when this scipt is run), so we just
# include a copy.
def get_version_from_git():
    try:
        p = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                             cwd=distr_root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return
    if p.wait() != 0:
        return
    # TODO: use os.path.samefile once we depend on Python >= 3.3.
    if os.path.normpath(p.communicate()[0].rstrip('\n')) != distr_root:
        # The top-level directory of the current Git repository is not the same
        # as the root directory of the Kwant distribution: do not extract the
        # version from Git.
        return

    # git describe --first-parent does not take into account tags from branches
    # that were merged-in.
    for opts in [['--first-parent'], []]:
        try:
            p = subprocess.Popen(['git', 'describe'] + opts, cwd=distr_root,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return
        if p.wait() == 0:
            break
    else:
        return
    version = p.communicate()[0].rstrip('\n')

    if version[0] == 'v':
        version = version[1:]

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=distr_root)
    except OSError:
        version += '-confused'  # This should never happen.
    else:
        if p.wait() == 1:
            version += '-dirty'
    return version
