import bz2
import io
import json
import os
import random
from glob import glob
from multiprocessing import Pool

import pandas as pd
import zstandard

reddit_directory = '/shared/2/datasets/reddit-dump-all/RC/'
output_directory = '/shared/3/projects/hiatus/multilingual/reddit-spanish/'
spanish_subreddits = {
    'r/Ciencia': 'Science topics.',
    'r/Cinefilos': 'Films from Spain and Latin America.',
    'r/Chistes': 'Jokes.',
    'r/ConsejosDePareja': 'Relationship advice for SOs.',
    'r/cuentaleareddit': 'Casual conversation.',
    'r/Desahogo': 'Sub for venting.',
    'r/espanol': 'The first subreddit in Spanish ever.',
    'r/filosofia_en_espanol': 'Philosophical discussions.',
    'r/fisica': 'News and popular science related to Physics.',
    'r/Futbol': 'World football (soccer) matters.',
    'r/HistoriasDeReddit': 'Random anecdotes and stories by the community members.',
    'r/HistoriasdeTerror': 'Horror stories,',
    'r/Jagverse': 'Technology and science topics.',
    'r/Latinos': 'Similar to the Latin American sub.',
    'r/Libros': 'Books and literature',
    'r/Programacion': 'Programmer community.',
    'r/preguntaleareddit': 'The equivalent to r/AskReddit.',
    'r/RedditPregunta': 'The equivalent to r/AskReddit.',
    'r/Redditores': 'Another multi-topic subreddit.',
    'r/relaciones': 'General relationship advice.',
    'r/Spanishhelp': 'The sister subreddit of r/Spanish. Homework questions, exercise checks and text proofreads can be requested here.',
    'r/Videojuego': 'Gaming community.',
    'r/webcomicsenespanol': 'Web comics.',
    'r/WriteStreakES': 'The equivalent to r/WriteStreak but in Spanish. The main goal is to constantly practice writing in a target language; the community offers corrections.',
    'r/yo_elvr': 'Generic "me_irl" memes.',
    'r/Argentina': None,
    'r/ArgenBeauty': 'Beauty tips in Argentina.',
    'r/ArgEntos': 'Cannabis ethusiasts from Argentina.',
    'r/ArgenCirclejerk': None,
    'r/dankgentina': 'Memes and shitposting.',
    'r/ForwardsDeMama': 'Similar to r/Forwardsfromgrandma.',
    'r/Fulbo': 'Football in Argentina.',
    'r/SquarePosting': 'Memes.',
    'r/Bolivia': None,
    'r/Chile': None,
    'r/LaRoja': 'National Chilean football soccer team.',
    'r/ChileCringe': None,
    'r/yo_ctm': 'Similar to r/yo_elvr but adapted for a Chilean audience.',
    'r/Colombia': None,
    'r/Ticos': '(Costa Rica)',
    'r/Dominican': None,
    'r/Ecuador': None,
    'r/EstadosUnidos': 'Spanish speaking people in the U.S.',
    'r/Latinoamerica': 'Latin American matters.',
    'r/Mexico': None,
    'r/Chairos': 'Low-effort Mexican liberalism and leftist activism on social media.',
    'r/FutbolMX': 'Football in Mexico.',
    'r/LigaMX': 'Mexican football league tournaments.',
    'r/MAAU': 'Memes and shitposts from mostly a Mexican audience.',
    'r/MeMexico': 'Mexican memes.',
    'r/MexicoCircleJerk': 'Self-explanatory',
    'r/Mujico': 'Memes and shitpostings.',
    'r/VideojuegosMX': 'Videogame scene.',
    'r/Panama': None,
    'r/Paraguay': None,
    'r/Peru': None,
    'r/PuertoRico': None,
    'r/Spain': None,
    'r/Asi_va_Espana': 'Memes and shitposting.',
    'r/es': 'Spanish (Spain) matters.',
    'r/LaLiga': 'Spanish football league tournament.',
    'r/mapassincanarias': 'Maps without the Canary Islands, akin to r/MapsWithoutNZ.',
    'r/SpainPolitics': None,
    'r/Uruguay': None,
    'r/Vzla': None,
    'r/Vencirclejerk': 'Circlejerk in Venezuela.',
}
english_subs = {'r/politics', 'r/ukpolitics', 'r/eupolitics', 'r/AtlantaUnited', 'r/tfc', 'r/WriteStreakEN',
                'r/offmychest', 'r/soccer', 'r/england', 'r/usa', 'r/programming', 'r/dankmemes', 'r/memes',
                'r/soccercirclejerk', 'r/nbacirclejerk'}


def process_zst_file(reddit_file):
    print(f"Processing {reddit_file}")
    dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
    comments = []
    with open(reddit_file, 'rb') as fh:
        try:
            reader = dctx.stream_reader(fh)
            stream = io.BufferedReader(reader)
            for line in stream:
                process_line(comments, line, reddit_file)
        except Exception as e:
            print(f"An error occurred while processing {reddit_file}: {str(e)}")

    if comments:
        save_subreddit_comments(reddit_file, comments)


def process_bz2_file(reddit_file):
    print(f"Processing {reddit_file}")
    comments = []
    with bz2.BZ2File(reddit_file, 'r') as f:
        for line in f:
            try:
                process_line(comments, line, reddit_file)
            except Exception as e:
                print(f"An error occurred while processing {reddit_file}: {str(e)}")
                return

    if comments:
        save_subreddit_comments(reddit_file, comments)


def process_line(comments, line, reddit_file):
    data = json.loads(line.decode('utf-8'))
    sub = data['subreddit'].lower()
    if sub in spanish_subs or sub in english_subs:
        language = 'spanish' if sub in spanish_subs else 'english'
        comments.append({
            'file': reddit_file,
            'author': data['author'],
            'subreddit': data['subreddit'],
            'created_utc': data['created_utc'],
            'link_id': data['link_id'],
            'parent_id': data['parent_id'],
            'text': data['body'],
            'language': language
        })

        # Intermittently save comments
        if len(comments) % 5000000 == 0:
            save_subreddit_comments(reddit_file, comments)


def save_subreddit_comments(reddit_file, subreddit_comments):
    try:
        df = pd.DataFrame(subreddit_comments)
        file_name = reddit_file.rsplit('/')[-1].replace('.zst', '') + '.parquet.gzip'
        file_path = os.path.join(output_directory, file_name)
        print(f"Saving {len(df)} rows to {file_path}")
        df.to_parquet(file_path, compression='gzip')
    except Exception as e:
        print(f"An error occurred while saving {reddit_file}: {str(e)}")


if __name__ == '__main__':
    spanish_subs = set([s[2:].lower() for s, _ in spanish_subreddits.items()])
    english_subs = set([e[2:].lower() for e in english_subs])
    old_reddit_files = glob(reddit_directory + '*.bz2')
    reddit_files = glob(reddit_directory + '*.zst')
    # Randomly select 3 older months and 6 newer months
    old_reddit_files = random.sample(old_reddit_files, 3)
    reddit_files = random.sample(reddit_files, 6)

    print(f"# of Spanish subreddits: {len(spanish_subs)}")
    print(f"# of English subreddits: {len(english_subs)}")
    print(f"# of Reddit files: {len(reddit_files)}")

    with Pool(len(reddit_files)) as pool:
        pool.map(process_zst_file, reddit_files)

    with Pool(len(old_reddit_files)) as pool:
        pool.map(process_bz2_file, old_reddit_files)
