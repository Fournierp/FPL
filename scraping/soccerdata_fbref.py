import soccerdata as sd
from pathlib import Path

fbref = sd.FBref(
    leagues="ENG-Premier League",
    seasons=[2022],
    no_cache=False,
    data_dir=Path('data/fbref_opta'))
epl_schedule = fbref.read_schedule(force_cache=True)
epl_schedule = epl_schedule[epl_schedule['game_id'].notna()]
epl_schedule = epl_schedule.game_id.apply(lambda x: fbref.read_shot_events(match_id=x, force_cache=True))
