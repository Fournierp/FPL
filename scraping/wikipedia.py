import os
import logging

import pandas as pd
import re


class Wikipedia:
    """Scrape Wikipedia"""

    def __init__(self, logger):
        """
        Args:
            logger (logging.logger): Logging package
        """
        self.root = 'data/wiki'
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.logger = logger

    def stadium_capacity(self):
        """ Scrape table of football stadium capacity """
        # URL Request
        df = pd.read_html(
            "https://simple.wikipedia.org/wiki/" +
            "List_of_English_football_stadiums_by_capacity")[0]
        # Remove empty row
        df = df[~(df.Capacity == "Other Listed Stadiums")]
        # Remove text
        df['Capacity'] = df['Capacity'].apply(
            lambda x: int(
                re.compile("\[\d*\]")
                .sub("", x)
                .replace(',', '')))
        # Save
        df.to_csv(
            os.path.join(self.root, "stadiums.csv"),
            index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger: logging.Logger = logging.getLogger(__name__)

    wiki = Wikipedia(logger)
    wiki.stadium_capacity()
