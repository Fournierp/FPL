import os
import logging

import pandas as pd
import requests
from bs4 import BeautifulSoup


class Hall_of_Fame:
    """Scrape livefpl"""

    def __init__(self, logger):
        """
        Args:
            logger (logging.logger): Logging package
        """
        self.root = 'data/hof'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.logger = logger

    def team_ids(self):
        """ Scrape table of top historical managers team IDs """
        # URL Request
        url = 'https://www.livefpl.net/elite'
        df = pd.read_html(url)[1]

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {"id": "main"})

        links = []
        for tr in table.findAll("tr"):

            trs = tr.findAll("td")
            for each in trs:
                try:
                    link = each.find('a')['href']
                    links.append(link[40:-8])
                except:
                    pass

        df['team_id'] = links

        # Save
        (
            df
            .drop('Unnamed: 1', 1)
            .to_csv(
                os.path.join(self.root, "top_managers.csv"),
                index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    hof = Hall_of_Fame(logger)
    hof.team_ids()
