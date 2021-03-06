import requests

import logging
import datetime


YML_FILE_HEAD = """on:
  workflow_dispatch:
  schedule:
"""

YML_FILE_FOOT = """
jobs:
  main:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run script
        env:
          BOT_GITHUB_ACCESS_TOKEN: ${{ secrets.BOT_GITHUB_ACCESS_TOKEN }}
        run: python {dir}/{script} foo bar"""


class Schedule:
    """Generate YML files with Cron date to run Github Actions"""

    def __init__(self, logger):
        """
        Args:
            logger (logging.logger): Logging package
        """
        self.deadlines = self.get_fpl_metadata()

        self.logger = logger

    def get_fpl_metadata(self):
        """ Request the FPL API

        Returns:
            (tuple): Next GW and player ids
        """
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()

        # Get deadlines
        deadlines = self.get_deadlines(res['events'])

        return deadlines

    def get_current_gw(self, events):
        """ Get the next gameweek to be played in the EPL

        Args:
            events (json): FPL API response

        Returns:
            (int): Next gameweek
        """
        for idx, gw in enumerate(events):
            if gw['is_current']:
                return idx + 1

    def get_deadlines(self, events):
        """ Convert FPL date to CRON times

        Args:
            events (json): FPL API response

        Returns:
            (list): List of dates
        """
        return [datetime.datetime.strptime(
            gw['deadline_time'], '%Y-%m-%dT%H:%M:%SZ') for gw in events]

    def schedule_github_actions(self):
        """Generate YML Files"""
        cron_job_template = '    - cron: \'{time}\'\n'

        # Scrape FiveThirtyEight a few hours before the deadline.
        script = 'fivethirtyeight'
        with open(f'.github/workflows/{script}.yml', 'w') as output_file:
            output_file.write(YML_FILE_HEAD)
            for gw, deadline in enumerate(self.deadlines):
                # Cronify deadlines
                delay = (deadline - datetime.timedelta(hours=12))
                cron_time = f'{delay.minute} {delay.hour} {delay.day} {delay.month} *'
                output_file.write(cron_job_template.format(time=cron_time))
            output_file.write(
                YML_FILE_FOOT.format(dir='scraping', script=f'{script}.py'))

        # Run score prediction model a few hours before the deadline.
        script = 'dixon_coles'
        with open(f'.github/workflows/{script}.yml', 'w') as output_file:
            output_file.write(YML_FILE_HEAD)
            for gw, deadline in enumerate(self.deadlines):
                # Cronify deadlines
                delay = (deadline - datetime.timedelta(hours=6))
                cron_time = f'{delay.minute} {delay.hour} {delay.day} {delay.month} *'
                output_file.write(cron_job_template.format(time=cron_time))
            output_file.write(
                YML_FILE_FOOT.format(dir='modeling', script=f'{script}.py'))

        script = 'bayesian_xg'
        with open(f'.github/workflows/{script}.yml', 'w') as output_file:
            output_file.write(YML_FILE_HEAD)
            for gw, deadline in enumerate(self.deadlines):
                # Cronify deadlines
                delay = (deadline - datetime.timedelta(hours=6))
                cron_time = f'{delay.minute} {delay.hour} {delay.day} {delay.month} *'
                output_file.write(cron_job_template.format(time=cron_time))
            output_file.write(
                YML_FILE_FOOT.format(dir='modeling', script=f'{script}.py'))

        # Scrape FPL Ownership a few hours after the deadline.
        script = 'fpl_gameweek'
        with open(f'.github/workflows/{script}.yml', 'w') as output_file:
            output_file.write(YML_FILE_HEAD)
            for gw, deadline in enumerate(self.deadlines):
                # Cronify deadlines
                delay = (deadline + datetime.timedelta(hours=3))
                cron_time = f'{delay.minute} {delay.hour} {delay.day} {delay.month} *'
                output_file.write(cron_job_template.format(time=cron_time))
            output_file.write(
                YML_FILE_FOOT.format(dir='scraping', script=f'{script}.py'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    schedule = Schedule(logger)
    schedule.schedule_github_actions()
