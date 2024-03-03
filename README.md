This is a python implementation of the "Fighting Chess Index" (FCI) defined by David Smerdon in this post, which is a measure of the combativeness of top chess players. With PGN files as input, the script calculates average Elo rating, average draw length, short draw rates (fewer than 32 moves) and draws with white for each player.

These draw metrics are reduced to a single score by PCA, which is then used as the dependent variable in a linear regression model with average Elo and Elo difference from opponents as predictors. The model's residuals represent combativeness deviation from Elo-based expectations. These residuals are inverted and rescaled between 0 and 100, producing the final FCI where higher values signify greater combativeness.


## Usage

If you want to filter by average Elo or require a minimum number of games, change these lines in `fci.py`:

```python
player_stats = player_stats[player_stats['eloav'] >= 2500] # change eloav to minimum elo

player_stats = player_stats[player_stats['N'] >= 1] # change N to minimum number of games
```

Replace `folder` with the path to your folder of PGN files in this line before running `fci.py`:

```python
pgn_files = glob.glob('folder/*.pgn')
```

