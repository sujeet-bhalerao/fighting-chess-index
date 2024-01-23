`fci.py` is a python script that implements David Smerdon's Fighting Chess Index from [this post](https://www.davidsmerdon.com/?p=2168) given a PGN file or a folder of PGN files.


## Usage

If you want to filter by average Elo or require a minimum number of games, change these lines in `fci.py`:

```python
player_stats = player_stats[player_stats['eloav'] >= 2500] # change eloav to minimum elo

player_stats = player_stats[player_stats['N'] >= 1] # change N to minimum number of games
```

Replace `folder` with the path to your folder of PGN files in this line before running the script:

```python
pgn_files = glob.glob('folder/*.pgn')
```

