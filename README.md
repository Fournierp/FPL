# Fantasy-Premier-League

## Introduction

Fantasy Premier League related analytics. In this repository, I conduct data analysis on the numerous facets of football and particularly apply it to the fantasy game.

## Installation

Firstly, ensure that you have pip install. In which case follow these steps using the command line:

```{bash}
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

Then install the required libraries listed in the requirements.txt

```{bash}
pip install -r requirements.txt
```

## Usage Example

```{bash}
python3 /home/FPL/modeling/dixon_coles.py
```

## Features

- Github Actions to scrape:
  - Daily betting odds
  - FiveThirtyEight SPIs before the games
  - FPL ownership data per gameweek
  - FPL Review Predictive data

- Optimization of FPL Team
- Predictions of match scores


## Release history

* 0.0
  * Data collections of managers season data through the Official Fantasy Premier League API.

* 1.0
  * Github Actions scraper
  * Basic gameweek FPL Team optimization
  * Benchmark match scores predictive models

## Acknowledgements

[FPL](https://fantasy.premierleague.com/) - Official data on player ownership, chips used etc.

[Football-Data.co.uk](https://www.football-data.co.uk/data.php) - Historical Football Results and Betting Odds Data

[FBref](https://fbref.com/en/) - Football Stats and History Statistics, scores and history

[FiveThirtyEight](https://projects.fivethirtyeight.com/soccer-predictions/premier-league/) - Forecasts and Soccer Power Index (SPI) ratings

[FPL Review](https://fplreview.com/) - FPL Predictions

[Logos](https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1) - Premier League clubs logos

[Understat](https://understat.com/league/EPL) - Shot Expected goals and locations


## Contribute

To build on this tool, please fork it and make pull requests. Or simply send me some suggestions !

## Authors

* **Paul Fournier** - *Initial work* - [Fournierp](https://github.com/Fournierp)
