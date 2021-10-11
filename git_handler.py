"""
Git module
"""
import os

import git


def git_push(*args, message=''):
    try:
        path = os.getcwd() + '/'
        repo = git.Repo(path)
        repo.git.add([path + file for file in args])
        repo.index.commit(message)
        origin = repo.remote('origin')
        origin.push()
    except Exception as e:
        print("git push error: ", e)
    else:
        print("git push done")
