
import chess.pgn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import glob

def parse_pgn_files(pgn_files):
    data = []
    for pgn_file in pgn_files:
        print("file: " + pgn_file)
        try:
            f = open(pgn_file, 'r', encoding='ISO-8859-1')
        except UnicodeDecodeError:
            f = open(pgn_file, 'r', encoding='utf-8', errors='ignore')
            print("UnicodeDecodeError")
        with f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                round_info = game.headers.get('Round', 'Unknown')
                white = game.headers['White']
                black = game.headers['Black']
                
                try:
                    white_elo = int(game.headers.get('WhiteElo', '0'))
                    black_elo = int(game.headers.get('BlackElo', '0'))
                except ValueError:
                    continue
                result = game.headers['Result']
                num_moves = len(list(game.mainline_moves()))
                is_draw = result == '1/2-1/2'
                is_short_draw = is_draw and num_moves <= 64

                data.append({
                    'round': round_info,
                    'player': white,
                    'result': result,
                    'elo': white_elo,
                    'opponent_elo': black_elo,
                    'is_draw': is_draw,
                    'is_short_draw': is_short_draw,
                    'num_moves': num_moves,
                    'color': 'white'
                })
                data.append({
                    'round': round_info,
                    'player': black,
                    'result': result,
                    'elo': black_elo,
                    'opponent_elo': white_elo,
                    'is_draw': is_draw,
                    'is_short_draw': is_short_draw,
                    'num_moves': num_moves,
                    'color': 'black'
                })
    #pd.DataFrame(data).to_csv('data.csv')
    return pd.DataFrame(data)

def compute_fci(df):
    player_stats = df.groupby('player').agg({
        'elo': 'mean',
        'is_draw': 'mean',
        'is_short_draw': 'mean',
        'num_moves': lambda x: np.mean(x[df['is_draw']]) if any(df['is_draw']) else 0
    }).rename(columns={'elo': 'eloav', 'is_draw': 'd', 'is_short_draw': 'd_short', 'num_moves': 'd_length'})
    player_stats['N'] = df.groupby('player').size()
    player_stats['d_short_W'] = df[(df['color'] == 'white') & (df['is_short_draw'])].groupby('player').size() / player_stats['N']
    player_stats['elodiff'] = df.groupby('player').apply(lambda x: np.mean(np.abs(x['elo'] - x['opponent_elo'])))

    player_stats = player_stats[player_stats['eloav'] >= 2500] # filter by min average elo
    player_stats = player_stats[player_stats['N'] >= 1] # filter by minimum number of games

    pca_vars = player_stats[['d', 'd_short', 'd_short_W', 'd_length']].fillna(0)
    pca = PCA(n_components=1)
    pca.fit(pca_vars)
    player_stats['pca_score'] = pca.transform(pca_vars).flatten()

    X = player_stats[['eloav', 'elodiff']]
    y = player_stats['pca_score']
    reg = LinearRegression().fit(X, y)
    player_stats['fci_residual'] = y - reg.predict(X)


    min_residual, max_residual = player_stats['fci_residual'].min(), player_stats['fci_residual'].max()
    player_stats['FCI'] = (max_residual - player_stats['fci_residual']) / (max_residual - min_residual) * 100
    #player_stats.to_csv('player_stats.csv')

    return player_stats.sort_values(by='FCI', ascending=False)


def display_game_results(df):
    results_df = df[['round', 'player', 'color', 'result', 'num_moves']]
    results_df = results_df.sort_values(by=['round', 'player'])
    print(results_df.to_string(index=False))


pgn_files = glob.glob('folder/*.pgn')
df = parse_pgn_files(pgn_files)
display_game_results(df)
fci_df = compute_fci(df)
fci_df.to_csv('fci.csv')
print(fci_df)
