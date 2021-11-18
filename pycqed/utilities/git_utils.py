import subprocess
import os


def git_commit(msg='auto backup commit',
	repo_dir=None, print_output=False, encoding='utf-8'):
	"""
	Runs "git commit -am 'msg'" in the specified repo directory

	Does no pull nor push
	"""
	stdout_str = None
	stderr_str = None
	cmds = ['git', 'commit', '-am', msg]
	process = subprocess.Popen(
		cmds,
		cwd=repo_dir,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT)
	stdout, stderr = process.communicate()
	if stdout is not None:
		stdout_str = stdout.decode(encoding)
	if stderr is not None:
		stderr_str = stderr.decode(encoding)
	if print_output:
		print('\n===========\nGIT COMMIT\n===========\nSTDOUT:\n{}\nSTDERROR:\n{}'.format(stdout_str, stderr_str))
	return stdout_str, stderr_str


def git_status(repo_dir=None, print_output=False, encoding='utf-8'):
	stdout_str = None
	stderr_str = None
	cmds = ['git', 'status']
	process = subprocess.Popen(
		cmds,
		cwd=repo_dir,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT)
	stdout, stderr = process.communicate()
	if stdout is not None:
		stdout_str = stdout.decode(encoding)
	if stderr is not None:
		stderr = stderr.decode(encoding)
	if print_output:
		print('\n===========\nGIT STATUS\n===========\nSTDOUT:\n{}\nSTDERROR:\n{}'.format(stdout_str, stderr_str))
	return stdout_str, stderr_str


def git_get_last_commit(author=None, repo_dir=None,
	print_output=False, encoding='utf-8'):
	stdout_str = None
	stderr_str = None
	cmds = ['git', 'log', '-n', '1']
	if author is not None:
		cmds = cmds + ['--author=' + author]
	process = subprocess.Popen(
		cmds,
		cwd=repo_dir,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT)
	stdout, stderr = process.communicate()
	if stdout is not None:
		stdout_str = stdout.decode(encoding)
	if stderr is not None:
		stderr = stderr.decode(encoding)
	if print_output:
		print('\n===============\nGIT LAST COMMIT\n===============\nSTDOUT:\n{}\nSTDERROR:\n{}'.format(stdout_str, stderr_str))

	return os.linesep.join(stdout_str.split(os.linesep)[:3]), stderr
