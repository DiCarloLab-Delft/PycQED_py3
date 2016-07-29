##Simple but useful tutorial explaining most of the git commands you will need
http://rogerdudler.github.io/git-guide/

##Branching
###To Create a new branch
* git branch -> make sure you are on the branch (master most of the time) from which you want to branch off
* git pull -> to make sure you are up to date before creating a new branch
* git checkout -b 'name of your branch' -> create your new branch and switch to it
* git push origin 'name of your branch' -> to make sure other people can see it on github

###Whenever you want to know which branch you are on use the command
* git branch -> list the available branches
* git checkout 'branchname' -> switch to the branch named 'branchname'

### Only pushing current branch instead of all branches with git push
* git config --global push.default current
This is important when working on a branch because otherwise git will complain that your master is behind when trying to push

### Closing an Issue
When you are done on a branch you have probably closed an issue.
By including the text close #23 in the commit message, the commit automatically closes issue 23 when pushed to github.

### Merging with master
After you are content with the changes you made (and have tested them) it is time to merge back into the master. The correct way to do this is as follows:

1. git pull origin master -> Pull in the changes on the master into the branch you are working on.
Note: pulling in the master will likely result in conflicts, resolve these conflicts and commit the changes
2.   git commit
3.   git push -> push the changes on the branch back to github.
4.   git checkout master -> make the master branch your active branch
5.   git pull -> pull in latest changes on the master
6.   git merge 'name of your branch'  -> merges the branch into the master
7.   git push -> you are done, the changes should be incorporated in the master

#### Resolving merge conflicts
If there is a conflict between two files there are two ways to resolve this. The preferred way is to perform a manual merge. If this is not possible it is possible to keep either version of the file.
##### Performing a manual merge
 open the conflicting file in sublime and look for "<<<<", "====" and ">>>>". These indicate the parts of the file that are conflicted. This can then be manually edited and saved. The merge can be completed by committing the updated files.
##### Keeping either version of a conflicted file
If a file (e.g. index.html) is conflicting you can find keep either your own (--ours) or the version you pulled in (--theirs) with the following commands.
```
git checkout --ours index.html
git checkout --theirs index.html
```

### Checking out a specific commit
* git log -> lists all the <sha1> hashes of previous commits
* git checkout 'hash' -> checkout out that specific hash

### Updating all submodules to the latest version of the code

```
git submodule foreach git pull origin PycQEDbranch
```


### Committing changes on a detached head branch to branch 'branchname'
Create a branch at the detached head state
* git branch temp
* git checkout temp
* git merge branchname


Ignoring files computer-specific
It is possible to ignore changes made to a file without having to add an entry to .gitignore. An example where this might be useful is with config files, where the path to directories may vary per computer. To do so enter the following command into your git shell:
   git update-index --assume-unchanged 'filename'
