import subprocess
import datetime


class Git:
    """ Class to push files to git from Github Action """

    def __init__(self):
        """Run all the steps"""
        self.set_config()
        self.git_add()
        self.git_commit()
        self.git_push()

    def _run(self, args):
        """ Run command line commands

        Args:
            args (array): Arguments for the CL
        """
        result = subprocess.run(
            args=args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result

    def set_config(self):
        """Set the configurations with my Github account"""
        config = ["git", "config", "--global"]
        config_name = self._run(config + ["user.name", "PF (Bot Account)"])
        config_email = self._run(
            config + ["user.email", "paul.ag.fournier@gmail.com"])

        if config_name.returncode != 0:
            raise Exception("Failed to configure name")
        if config_email.returncode != 0:
            raise Exception("Failed to configure email")

    def git_add(self):
        """Git add files"""
        add = self._run(["git", "add", "-A"])
        if add.returncode != 0:
            raise Exception("Failed to stage changes")

    def git_commit(self):
        """Git commit files"""
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        message = "Git Action: {}".format(now_str)
        commit = self._run(["git", "commit", "-m", message])
        if commit.returncode != 0:
            raise Exception("Failed to commit changes")

    def git_push(self):
        """Git push files"""
        push = self._run(["git", "push", "origin", "main"])
        if push.returncode != 0:
            raise Exception("Failed to push changes")
