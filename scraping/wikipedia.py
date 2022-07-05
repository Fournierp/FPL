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

    #PL Handbook 2021-22: https://resources.premierleague.com/premierleague/document/2021/07/22/1107e7b4-9c37-4483-9cc1-292a3ed53e6f/PL_Handbook_2021_22_DIGITAL_22.07.pdf
    #PL Handbook 2020-21: https://resources.premierleague.com/premierleague/document/2020/09/11/dc7e76c1-f78d-45a2-be4a-4c6bc33368fa/2020-21-PL-Handbook-110920.pdf
    #PL Handbook 2019-20: https://resources.premierleague.com/premierleague/document/2020/07/24/70ec483e-7207-42cd-89d9-576e53befedd/2019-20-PL-Handbook-240720.pdf
    #PL Handbook 2018-19: https://resources.premierleague.com/premierleague/document/2018/07/30/8944eb84-6450-4f80-8f46-64c6e2e1b929/PL_Handbook-2018-19.pdf
    #PL Handbook 2017-18: https://premierleague-static-files.s3.amazonaws.com/premierleague/document/2017/08/11/c494a26e-b573-41e4-bcd2-daf0ca76a00d/PL_Handbook_2017-18_Digital-4-.pdf
    #PL Handbook 2016-17: https://resources.premierleague.com/premierleague/document/2016/08/09/b81992f4-cf2a-4c5e-a9c1-155169074163/2016-17_Premier_League_Handbook.pdf