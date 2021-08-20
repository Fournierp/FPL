import requests

import logging
import datetime
import time

import subprocess


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
        run: python scraping/{script} """


class Schedule:

    def __init__(self, logger):
        
        self.deadlines = self.get_fpl_metadata()

        self.logger = logger


    def get_fpl_metadata(self):
        url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
        res = requests.get(url).json()
                        
        # Get deadlines
        deadlines = self.get_deadlines(res['events'])

        return deadlines


    def get_current_gw(self, events):
        for idx, gw in enumerate(events):
            if gw['is_current']:
                return idx + 1


    def get_deadlines(self, events):
        return [datetime.datetime.strptime(gw['deadline_time'], '%Y-%m-%dT%H:%M:%SZ') for gw in events]


    def schedule_github_actions(self):
        cron_job_template = '    - cron: \'{time}\'\n'

        # Scrape FiveThirtyEight a few hours before the deadline.
        script='fivethirtyeight'
        with open(f'.github/workflows/{script}.yml', 'w') as output_file:
            output_file.write(YML_FILE_HEAD)
            for gw, deadline in enumerate(self.deadlines):
                # Cronify deadlines
                delay = (deadline - datetime.timedelta(hours=12))
                cron_time = f'{delay.minute} {delay.hour} {delay.day} {delay.month} *'
                output_file.write(cron_job_template.format(time=cron_time))
            output_file.write(YML_FILE_FOOT.format(script=f'{script}.py'))

        # Scrape FPL Ownership a few hours after the deadline.
        script='fpl_gameweek'
        with open(f'.github/workflows/{script}.yml', 'w') as output_file:
            output_file.write(YML_FILE_HEAD)
            for gw, deadline in enumerate(self.deadlines):
                # Cronify deadlines
                delay = (deadline + datetime.timedelta(hours=3))
                cron_time = f'{delay.minute} {delay.hour} {delay.day} {delay.month} *'
                output_file.write(cron_job_template.format(time=cron_time))
            output_file.write(YML_FILE_FOOT.format(script=f'{script}.py'))

        # website='fpl_review'
        # with open(f'.github/workflows/{website}.yml', 'w') as output_file:
        #     output_file.write(YML_FILE_HEAD)
        #     for gw, deadline in enumerate(self.deadlines):
        #         # Cronify deadlines
        #         eight_hour_prior = (deadline - datetime.timedelta(hours=8))
        #         cron_time = f'{eight_hour_prior.minute} {eight_hour_prior.hour} {eight_hour_prior.day} {eight_hour_prior.month} *'
        #         output_file.write(cron_job_template.format(time=cron_time))
        #     output_file.write(YML_FILE_FOOT.format(script=f'{website}.py'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    schedule = Schedule(logger)
    schedule.schedule_github_actions()