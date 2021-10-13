"""
Git module
"""
import os

import git

from settings import OS_CON


def git_push(*args, message="", update_db=False):
    try:
        path = os.getcwd() + OS_CON
        repo = git.Repo(path)
        repo.git.add([path + file for file in args])
        repo.index.commit(message)
        origin = repo.remote("origin")
        origin.push("master")
    except Exception as e:
        print("git push error: ", e)
    else:
        print("git push done")


def git_pull():
    repo = git.Repo(os.getcwd() + OS_CON)
    origin = repo.remotes.origin
    origin.pull()

    return repo.head.reference
