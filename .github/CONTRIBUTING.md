# Contributing
Thanks for your interest in the project! We welcome pull requests from developers of all skill levels.

If you find a bug or want to propose a new feature open an issue. If you have written some code that should be merged open a pull request describing your changes and why it should be merged.
If you have a question or want to discuss something, feel free to send an email to Adriaan Rol (m.a.rol@tudelft.nl).
## Git branching model

The  branching model is based on the branching model described in the post [A successful Git branching model](http://nvie.com/posts/a-successful-git-branching-model/).

### The main branches

**!!! To be discussed **
The central repo holds two main branches with an infinite lifetime:

* master
* development

We consider _origin/master_ to be the main branch where the source code of HEAD always reflects a production-ready state.

We consider _origin/develop_ to be the main branch where the source code of HEAD always reflects a state with the latest delivered development changes for **the next release**. Some would call this the “integration branch”. This is where any automatic nightly builds are built from. When the source code in the _develop_ branch reaches a stable point and is ready to be released, all of the changes should be merged back into _master_ somehow and then tagged with a release number. Therefore, each time when changes are merged back into _master_, this is a new production release **by definition**.

### Supporting branches
Next to the main branches _master_ and _development_ we use the following supporting branches.

* Task branches
* Bug branches
* Hotfix branches

Unlike the main branches, these branches have a limited life time, since they will be removed eventually

## Git Issue labeling system
Every issue and pull request should be labeled with two required labels, type and priority, and optional project dependent extra labels. Optional extra labels exist to further specify what category an issue belong to
 
### Issue type labels
- **type:task**: indicates that the issue concerns a task. Tasks are used in the GitHub project Kanban board.
- **type:bug**: indicates that the issue concerns a bug discoverd when testing or using the concerned project.
- **type:hotfix**: indicates that the issue concerns a bug in a live system (see description above related to the Hotfix branch). Normaly the issue will be reported as a *type:bug* and is lateron promoted to a *type:hotfix*.
- **type:enhancement**: indicates that the issue concerns a request for or a suggestion to an enhancement to an existing feature.
- **type:feature**: indicates that the issue concerns a request for a new feature.
- **type:question**: indicates that the issue concerns a question about e.g. the use of the system or a software funtion.
An issue should have only one type label assigned. A type label is not supposed to change. The *type:bug* to *type:hotfix* change is an exeption to this rule.

### Issue priority labels
- **prio:high**: the issue is blocking and **must** be solved on short notice.
- **prio:medium**: the issue is importent and **should** be solved or implemented as soon as possible.
- **prio:low**: the issue is not urgent and **can** be solved when time allows.
- **prio:very low**: the issue is something to consider for e.g. future releases.
An issue can have only one priority type label. A priority label can change over time. 

### Issue category labels (DDM project related)
Optional extra labels exist to further specify what category an issue belong to. These are repository dependent and prefixed with "Cat:

An issue can have multiple category labels assigned.
