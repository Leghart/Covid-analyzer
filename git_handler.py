"""
Git module
"""
import os

import git


def git_push(*args, message='', update_db=False):
    try:
        path = os.getcwd() + '/'
        repo = git.Repo(path)
        repo.git.add([path + file for file in args])
        repo.index.commit(message)
        origin = repo.remote('origin')
        print("Branch: ", repo.active_branch)
        if update_db:
            branch = "master"
        else:
            branch = repo.active_branch
        print('selected branch : ', branch)
        origin.push(branch)
    except Exception as e:
        print("git push error: ", e)
    else:
        print("git push done")
