# Contributing
Thanks for your interest in the project! We welcome pull requests from developers of all skill levels.

If you find a bug or want to propose a new feature open an issue. If you have written some code that should be merged open a pull request describing your changes and why it should be merged.
If you have a question or want to discuss something, feel free to send an email to Adriaan Rol (m.a.rol@tudelft.nl).

## Git branching model

The branching model is based on the branching model described in the post [A successful Git branching model](http://nvie.com/posts/a-successful-git-branching-model/), but differs in that the development branch is replaced by several "project" branches. 

### The main branches

The central repository holds three types of branches:

* master 
* project branches
* supporting branches

#### The master branch 
We consider _origin/master_ to be the main branch where the source code of HEAD always reflects a production-ready state.

* safe to merge into project branches
* staring point for supporting branches
* contains only reviewed code

#### The project branch
Every project/experiment has it's own project branch, prefixed with _Proj/_. We consider the project branch to be the main branch where the source code of HEAD always reflects a state with the latest changes used in the project. The project branches themselves should never get merged into the master. Rather, individual changes should be cherry-picked and merged in the master using discrete pull requests. See also *merging strategy* below.

* prefixed with *Proj/*
* never get merged into master
* changes get cherry-picked into supporting branches to merge into master 


#### Merging strategy and supporting branches
Supporting branches are used to develop and integrate new code for project branches and integrate it into the master. 

Supporting branches should be used for a single change and should ideally correspond to a single issue. Supporting branches should be prefixed with the type of issue they address, see **issue type labels** below for details. Once the code on the branch is ready for review, a pull request into the master should be opened. 

**Supporting branches**

* correspond to a single issue
* prefixed with the issue type
* merged into master through a pull request

** Pull request**
In order for the pull request to be merged, the following conditions must be met:

* travis test suite passes
* all reasonable issues raised by codacy must be resolved 
* a positive review is required

Whenever possible the pull request should:

* follow the PEP8 style guide 
* have tests for the code
* be well documented and contain comments

Tests are not mandatory as this is generally hard to make for instruments that interact with hardware. 

If you want to get changes from a project branches onto a supporting branch for a pull request consider using the following snippet. 
```
git checkout master
git pull
git checkout -b 'branch name'
git cherry-pick #commitID
git push --set-upstream origin 'branch name'
```
Don't forget to switch back to your project branch after cherry picking
```
git checkout 'projectbranch'
```

## Git Issue labeling system
Every issue and pull request should be labeled with two required labels, type and priority, and optional project dependent extra labels. Optional extra labels exist to further specify what category an issue belong to
 
### Issue type labels
- **type:task**: indicates that the issue concerns a task. Tasks are used in the GitHub project Kanban board.
- **type:bug**: indicates that the issue concerns a bug discovered when testing or using the concerned project.
- **type:hotfix**: indicates that the issue concerns a bug in a live system (see description above related to the Hotfix branch). Normally the issue will be reported as a *type:bug* and is later on promoted to a *type:hotfix*.
- **type:enhancement**: indicates that the issue concerns a request for or a suggestion to an enhancement to an existing feature.
- **type:feature**: indicates that the issue concerns a request for a new feature.
- **type:question**: indicates that the issue concerns a question about e.g. the use of the system or a software function.
An issue should have only one type label assigned. A type label is not supposed to change. The *type:bug* to *type:hotfix* change is an exception to this rule.

### Issue priority labels
- **prio:high**: the issue is blocking and **must** be solved on short notice.
- **prio:medium**: the issue is important and **should** be solved or implemented as soon as possible.
- **prio:low**: the issue is not urgent and **can** be solved when time allows.
- **prio:very low**: the issue is something to consider for e.g. future releases.
An issue can have only one priority type label. A priority label can change over time. 

### Issue category labels 
Optional extra labels exist to further specify what category an issue belong to. These are repository dependent and prefixed with "Cat:

An issue can have multiple category labels assigned.


## Weekly code cleanup
In order to prevent diverging branches and as a general good practice there is a weakly code cleanup moment. 

During the weekly code cleanup you should: 

* Commit your code 
* Update your project branch with the latest changes from the master 
* Open pull requests containing progress of last week 
* review pull requests you have been asked to review 

It is recommended to do this more than once a week.
