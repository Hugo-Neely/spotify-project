import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time
import datetime
from typing import Union, Tuple
import numpy as np
import requests
from io import BytesIO
import matplotlib.dates as m_dates
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import json

# ensure data directory is set up as expected
DATA_DIR = os.path.abspath('data')
IMG_DIR = os.path.join(DATA_DIR, 'imgs')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger_csv_formatter = logging.Formatter('%(asctime)s.%(msecs)d,%(message)s', datefmt='%Y-%m-%d %H:%M:%S', style ='%')
logger_console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# tracking API calls to CSV
logger_calls_fh = logging.FileHandler(os.path.join(LOG_DIR, 'spotify_api_calls.csv'))
if not os.path.exists(os.path.join(LOG_DIR, 'spotify_api_calls.csv')):
    with open(os.path.join(LOG_DIR, 'spotify_api_calls.csv'), 'w') as f:
        f.write('time,message\n')
logger_calls_fh.setLevel(logging.DEBUG)
logger_calls_fh.set_name('spotify_api_calls')
logger_calls_fh.setFormatter(logger_csv_formatter)
logger_calls_fh.addFilter(lambda record: record.levelname == 'DEBUG')
logger.addHandler(logger_calls_fh)

# tracking other actions to file
logger_fh = logging.FileHandler(os.path.join(LOG_DIR, 'playlist_handler_info.log'))
if not os.path.exists(os.path.join(LOG_DIR, 'playlist_handler_info.log')):
    with open(os.path.join(LOG_DIR, 'playlist_handler_info.log'), 'w') as f:
        f.write('time,message\n')
logger_fh.setLevel(logging.INFO)
logger_fh.set_name('playlist_handler_info')
logger_fh.setFormatter(logger_console_formatter)  # use same formatter as console
logger.addHandler(logger_fh)

# logging warnings to console
# logger_ch = logging.StreamHandler()
# logger_ch.setLevel(logging.WARNING)
# logger_ch.set_name('console')
# logger_ch.setFormatter(logger_console_formatter)
# logger.addHandler(logger_ch)

supergenre_lists = {
    # EDM//electronic//dance
    'EDM':
    [
        'acid house',
        'alternative dance',
        'afro house', # afrobeats?
        'afro tech',
        'ballroom vogue',
        'baltimore club',
        'bass house',
        'bassline',
        'big beat',
        'breakbeat',
        'breakcore',
        'chicago house',
        'chillstep',
        'dance',
        'dancehall',
        'deep house',
        'disco house',
        'disco', # DISCO - could just be part of above?
        'downtempo',
        'drum and bass',
        'drumstep',
        'dub techno',
        'dubstep',
        'ebm',
        'edm',
        'edm trap',
        'electro',
        'electro swing', # lol
        'electroclash',
        'electronic',
        'electronica',
        'eurodance',
        'footwork',
        'freestyle',
        'french house',
        'funky house',
        'future bass',
        'g-house',
        'glitch',
        'hard house',
        'hard techno',
        'hi-nrg',
        'house',
        'idm', # 'intelligent dance music'. strong candidate for wankiest genre name
        'indie dance',
        'industrial',
        'italo dance',
        'italo disco',
        'jazz house',
        'jersey club',
        'jungle',
        'lo-fi house',
        'minimal techno',
        'moombahton',
        'new rave',
        'nu disco',
        'post-disco',
        'rally house',
        'stutter house',
        'synthwave',
        'tech house',
        'techno',
        'trance',
        'tropical house',
        'uk garage',
    ],

    'JAZZ':
    [
        'acid jazz',
        'bebop',
        'brazilian jazz',
        'cool jazz',
        'ethiopian jazz',
        'free jazz',
        'french jazz',
        'hard bop',
        'indie jazz',
        'jazz',
        'jazz fusion',
        'nu jazz',
        'smooth jazz',
        'swing music',
        'vocal jazz',
    ],

    'ROCK':
    [
        'acid rock',
        'alternative rock',
        'art rock',
        'anatolian rock',
        'aor', # 'album oriented rock'
        'argentine rock',
        'blues rock',
        'brazilian rock',
        'classic rock',
        'country rock',
        'deathrock',
        'folk rock',
        'garage rock',
        'glam rock',
        'gothic rock',
        'hard rock',
        'indie rock',
        'industrial rock',
        'krautrock',
        'lovers rock',
        'madchester',
        'math rock',
        'neo-psychedelic',
        'neue deutsche welle',
        'new wave',  # terribly defined genre. music that links punk and post punk. includes the jam, talking heads, ian dury. rock feels closest
        'noise rock',
        'post-hardcore',
        'post-rock',
        'progressive rock',
        'psychedelic rock',
        'rock',
        'rock and roll',
        'rockabilly',
        'roots rock',
        'slowcore',
        'soft rock',
        'southern rock',
        'space rock',
        'stoner rock',
        'surf rock',
        'yacht rock',
    ],

    'MISC':
    [
        'adult standards',
        'ambient',
        'avant-garde',
        'big band',
        'celtic', # better place for this? only really affects one song so not a huge deal
        'chanson',  # french lyric-driven
        'christmas', # lol
        'comedy',
        'easy listening',
        'exotica',  # ??
        'experimental',
        'german indie',
        'hardcore',  # too poorly defined to be useful. includes hardcore punk, hiphop, and the specific subgenre of edm that just goes by hardcore
        'indian indie',
        'indie',
        'italian singer-songwriter',
        'jam band',
        'japanese indie',
        'lo-fi indie',
        'lounge',
        'maluku',  # catch all for indonesian music (specifically from maluku islands)
        'musicals',
        'singer-songwriter',
        'soundtrack',
        'spoken word',
        'vaporwave', # idk where else to put this
        'variété française',
        'worship',
    ],

    'AFRICAN':
    [
        'afro adura',
        'afrobeat',
        'afrobeats',
        'afropiano',
        'afropop',
        'afroswing',
        'alté',
        'amapiano',
        'asakaa',  # ghanaian drill? in my playlists it only appears on one boj song, due to a feature
        'azonto', # ghanaian dance/hiphop. 
        'bikutsi',
        'hiplife',  # ghanaian (hiphop/ghanaian highlife apparently)
        'gnawa', # morrocan religious songs
        'highlife',
        'raï',  # algerian
        'rumba congolaise',  # rep. congo/drc dance
    ],

    'FOLK/COUNTRY':
    [   
        # COUNTRY
        'alt country',
        'classic country',
        # FOLK
        'anti-folk',
        'ambient folk',
        'americana', # i guess??
        'bluegrass',
        'folk',
        'newgrass',
        'traditional folk',
    ],

    'SOUL':
    [
        'afro soul',
        'cajun',  # not really but i have to put it somewhere
        'classic soul',
        'gospel',  # lol yes ok
        'indie soul',
        'motown',
        'northern soul',
        'philly soul',
        'retro soul',
        'soul',
        'soul blues',
        'soul jazz',
        'southern gospel',
    ],
    'BLUES':
    [
        'blues',
        'boogie-woogie',
        'zydeco',
        'classic blues',
        'country blues',
        'doo-wop',
        'jazz blues',
        'modern blues',
    ],

    'HIP-HOP':
    [
        'alternative hip hop',
        'aussie drill',
        'boom bap',
        'chinese hip hop',
        'cloud rap',
        'crunk',
        'drill',
        'east coast hip hop',
        'emo rap',
        'experimental hip hop',
        'french rap',
        'g-funk',
        'gangster rap',
        'german hip hop',
        'ghanaian hip hop',
        'grime',
        'hardcore hip hop',
        'hip hop',
        'hip house',
        'horrorcore',
        'hyphy',
        'jazz beats',
        'jazz rap',
        'lo-fi', # need to check this one TODO
        'lo-fi beats',
        'lo-fi hip hop',
        'melodic rap',
        'mexican hip hop',
        'miami bass',
        'midwest emo',
        'new orleans bounce',
        'nigerian drill',
        'old school hip hop',
        'rap',
        'rap rock',
        'sexy drill',
        'southern hip hop',
        'trap soul',
        'trip hop',
        'uk drill',
        'uk grime',
        'underground hip hop',
        'west coast hip hop',
    ],

    'R&B':
    [
        'afro r&b',
        'alternative r&b',
        'contemporary r&b',
        'dark r&b',
        'french r&b',
        'gospel r&b',
        'indie r&b',
        'neo soul',
        'quiet storm',
        'r&b',
        'uk r&b',
    ],

    'POP':
    [
        'art pop',
        'baroque pop',
        'bedroom pop',
        'brazilian pop',
        'britpop',
        'chillwave', # loose genre idk
        'dream pop',
        'electropop',
        'flamenco pop',
        'french indie pop',
        'french pop',
        'german indie pop',
        'german pop',
        'hyperpop',
        'indie pop',
        'jangle pop',
        'nederpop', # dutch pop
        'new jack swing',
        'pop',
        'pop soul',
        'pop urbaine',
        'power pop',
        'retro pop',
        'schlager',  # european pop that makes u smile
        'synthpop',
    ],

    'LATIN':
    [
        'axé',
        'bossa nova',
        'bolero', # spanish (rita payes)
        'candombe',  # uruguayan 
        'cha cha cha', # cuban 1950s dance
        'chicha',  # peruvian 60s
        'cumbia', # colombian folk(?). or maybe mexican
        'cumbia sonidera',  # mexican cumbia
        'electrocumbia',
        'fado',  # portuguese traditional? mournful
        'flamenco',
        'latin',
        'latin alternative',
        'latin folk',
        'latin folklore',
        'latin hip hop',
        'latin indie',
        'latin jazz',
        'latin pop',
        'latin rock',
        'mariachi',
        'merengue',  # dominican republic. dance? in 2/4
        'mexican indie',
        'mpb', # musican popular brasileira 🇧🇷
        'música mexicana',
        'nova mpb',
        'pagode', # brazillian 70s/80s
        'ranchera', # traditional mexican
        'salsa',
        'samba',
        'son cubano',
        'tango',
        'tejano',
        'trova',  # cuban
        'villancicos',  # spanish/portuguese folk
    ],

    'METAL':
    [
        'black metal',
        'djent',
        'drone metal',
        'glam metal',
        'progressive metal',
        'sludge metal',
    ],

    'CARRIBEAN':
    [
        'calypso',
        'dub',
        'ragga',
        'reggae',
        'rocksteady', # jamaican 60s
        'roots reggae',
        'ska',  # i guess? none of the ska i listen to is very carribean but it is a carribean genre lol
    ],

    'PUNK':
    [
        'celtic punk',
        'cold wave',
        'darkwave',
        'egg punk',
        'emo',
        'folk punk',
        'hardcore punk',
        'horror punk',
        'indie punk',
        'mathcore',  # v metaly but im putting it in here to boost punks numbers because i prefer that genre hehe
        'post-punk',
        'proto-punk',
        'punk',
        'queercore',
        'riot grrrl',
        'ska punk',
    ],

    'EAST ASIAN':
    [
        # realistically just japan
        'anime',
        'city pop',
        'j-pop', # these are all pushing it, but i feel like the songs they represent have a distinct enough sound to justify a separate group
        'j-r&b',
        'j-rap',
        'j-rock',
        'kayokyoku',
        'shibuya-kei',
    ],

    'CLASSICAL':
    [
        'classical',
        'classical piano',
        'medieval',
        'opera',
        'orchestral',
    ],

    'FUNK':
    [
        'funk',
        'funk melody',
        'funk pop',
        'funk rock',
        'jazz funk',
        'liquid funk',
        'uk funky',
    ]
}

supergenre_map = {}
for genre, lst in supergenre_lists.items():
    for subgenre in lst:
        supergenre_map[subgenre] = genre

class MonthlyPlaylistHandler:
    '''
    Handler for monthly playlist data. Contains a spotipy Spotify instance,
    handling authentication.
    '''
    
    data_dir = DATA_DIR

    supergenre_lists = supergenre_lists
    supergenre_map = supergenre_map

    
    def __init__(self, spotify_client:spotipy.Spotify = None):
        '''
        Create a new instance of the handler.

        Parameters
        ----------
        spotify_client : spotipy.Spotify, optional
            The spotify client handler from spotipy. Should be instantiated with
            a SpotifyOAuth manager as the auth_manager
        '''
        if spotify_client is None:
            self.spotify_client = self._new_spotify_client()
        elif isinstance(spotify_client, LoggingSpotifyClient):
            self.spotify_client = spotify_client
        elif isinstance(spotify_client, spotipy.Spotify):
            self.spotify_client = LoggingSpotifyClient(spotify_client = spotify_client)
        else:
            raise ValueError(f'Expected spotify_client to be of type spotipy.Spotify, got {type(spotify_client)}.')

    @staticmethod
    def _new_spotify_client(client_id:str = None, client_secret:str = None, redirect_uri:str = None,
                       backoff_factor:float = 0.5, **kwargs) -> spotipy.Spotify:
        '''
        Create a new instance of spotipy.Spotify(), optionally providing alternative
        authentication credentials.

        Parameters
        ----------
        client_id : str, optional
            client_id for Spotify API OAuth authentication. If None, the environment
            variable 'SPOTIPY_CLIENT_ID' will be used.
        client_secret : str, optional
            client_secret for Spotify API OAuth authentication. If None, the environment
            variable 'SPOTIPY_CLIENT_SECRET' will be used.
        redirect_uri : str, optional
            redirect_uri for Spotify API OAuth authentication. If None, the environment
            variable 'SPOTIPY_REDIRECT_URI' will be used.
        backoff_factor : float
            A factor to apply between API call attempts after the second try. 
            See https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html.
        **kwargs
            Optional keyword arguments to pass to spotipy.Spotify.
        '''

        if client_id is None:
            client_id = os.getenv('SPOTIPY_CLIENT_ID')
        if client_secret is None:
            client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
        if redirect_uri is None:
            redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')
        
        return LoggingSpotifyClient(
            auth_manager=spotipy.SpotifyOAuth(
                scope="playlist-read-private", 
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri
            ),
            backoff_factor=backoff_factor,
            **kwargs
        )
    
    @property
    def year_styles(self):
        with open(os.path.join(self.data_dir, 'year_styles.json', 'r')) as f:
            return json.load(f)
        
    def get_monthly_playlists(self, to_csv:bool = True, date_csv:bool = False) -> pd.DataFrame:
        '''
        Get a dataframe containing monthly playlists metadata.

        Parameters
        ----------
        to_csv : bool
            If True, will save the resulting pd.DataFrame as a CSV in the data directory.
            Filenames are in the format 'playlists_YYYY-MM-DD.csv'. Note that this means any
            other files created on this day will be overwritten.
        date_csv : bool
            If True, the saved CSV will have the date of data collection in the filename. Set to
            True to avoid overwriting previous files.
        '''

        got_all_pls = False
        offset = 0
        df = pd.DataFrame(columns=['collaborative', 'description', 'external_urls', 'href', 'id', 'images',
                                   'name', 'owner', 'primary_color', 'public', 'snapshot_id', 'tracks',
                                   'type', 'uri'])
        while not got_all_pls:
            playlists = self.spotify_client.current_user_playlists(limit = 50, offset = offset)
            df = pd.concat([df, pd.DataFrame(playlists['items'])], ignore_index = True)
            
            if len(playlists['items']) < 50:
                got_all_pls = True
            offset += 50

        ind = df['name'].astype(bool)  # series containing indexes of all monthly playlists
        ind[:] = False  # ensure we start with all False

        for year_style in self.year_styles.values():
            ind = ind | df['name'].str.contains(year_style)
        df = df.loc[ind]  # apply index

        df['n_tracks'] = df['tracks'].apply(lambda x: x['total'])

        month_to_int = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02',
            'mar': '03', 'march': '03',
            'apr':'04', 'april':'04',
            'may':'05',
            'jun':'06', 'june':'06',
            'jul':'07', 'july':'07',
            'aug':'08', 'august':'08',
            'sep':'09', 'september':'09',
            'oct':'10', 'october':'10',
            'nov':'11', 'november':'11',
            'dec':'12', 'december':'12'
        }
        
        # construct date
        df['date'] = df['name'].str.replace(' ', '-').str.casefold()
        for mth_name, mth_int in month_to_int.items():
            df['date'] = df['date'].str.replace(mth_name+'-', mth_int+'-')
        df['date'] = df['date'].str.replace('-22','-2022')
        df['date'] = df['date'].str.replace('-23','-2023')
        df['date'] = df['date'].str.replace('-24','-2024')
        df['date'] = df['date'].apply(lambda x: x[:7])
        df['date'] = pd.to_datetime(df['date']).dt.date

        # extract image url
        df['cover_image_url'] = df['images'].apply(lambda x: x[0]['url'] if not pd.isna(x[0]) else x)

        # rename href to url
        df = df.rename(columns={'href':'url'})

        # only keep useful columns
        df = df[['id', 'date', 'name', 'description', 'n_tracks', 'url', 'cover_image_url', 'snapshot_id']].set_index('id')

        # save if requested, and return
        if to_csv:
            if date_csv:
                filename = f'playlists_{str(datetime.datetime.now().date())}.csv'
            else:
                filename = 'playlists.csv'
            df.to_csv(
                os.path.join(self.data_dir, filename)
            )
        return df

    @property
    def latest_playlists_file(self) -> str:
        
        if self.playlists_base_csv_exists:
            return 'playlists.csv'
        
        # get the most recent date by extracting the date part of each playlist_DATE.csv file.
        # then convert to datetime, and get the most recent (max)
        most_recent_date = pd.to_datetime([file.replace('playlists_','').replace('.csv','') for file in os.listdir(self.data_dir) if 'playlists' in file]).max()

        return f'playlists_{str(most_recent_date.date())}.csv'

    @property
    def playlists_base_csv_exists(self) -> bool:
        return os.path.exists(os.path.join(self.data_dir, 'playlists.csv'))

    def read_monthly_playlists(self, date = None, download_if_required = False) -> pd.DataFrame:
        '''
        Read a saved DataFrame of monthly playlists.

        Parameters
        ----------
        date : str or datetime.date
            The date the data was collected. If None, will attempt to read the undated file,
            which is assumed to be the most recent. If the undated file does not exist,
            will read the most recent dated file.
            If the date is specified as a string, it should be given in the form YYYY-MM-DD.

        download_if_required : bool
            If True, will download the data using `get_monthly_playlists()` if the requested
            file does not exist.
            If False (default), will raise a FileNotFoundError if the requested file does not exist.

        Returns
        -------
        pandas.DataFrame
        '''
        
        if date is None:
            file = self.latest_playlists_file
        elif isinstance(date, datetime.date):
            file = f'playlists_{str(date.date())}.csv'
        elif isinstance(date, str):
            file = f'playlists_{date}.csv'
        else:
            raise ValueError(f'Unexpected type "{type(date)}" for date input encountered. Try entering date as either a string (YYYY-MM-DD) or datetime.date.')
        
        if not os.path.exists(os.path.join(self.data_dir, file)):
            if download_if_required:
                if date is None:
                    df = self.get_monthly_playlists(to_csv=True)
                else:
                    raise FileNotFoundError(f'File {file} does not exist in data directory {self.data_dir}. Cannot download data for a specific date.')
            else:
                raise FileNotFoundError(f'File {file} does not exist in data directory {self.data_dir}. Set download_if_required=True to download the data.')
        else:
            df = pd.read_csv(
                os.path.join(self.data_dir, file), 
            ).set_index('id')
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        return df
    
    def playlist_id(self, playlist_date:Union[str, datetime.date]) -> str:
        '''
        Get the Spotify ID of a monthly playlist from its date.

        Parameters
        ----------
        playlist_date : str or datetime.date
            The date of the playlist. If a string, should be in the form YYYY-MM-DD.

        Returns
        -------
        str
            The Spotify ID of the playlist.
        '''
        if isinstance(playlist_date, str):
            playlist_date = pd.to_datetime(playlist_date).date()
        elif not isinstance(playlist_date, datetime.date):
            raise ValueError(f'Unexpected type "{type(playlist_date)}" for date input encountered. Try entering date as either a string (YYYY-MM-DD) or datetime.date.')

        df_pl = self.read_monthly_playlists()
        try:
            pl_id = df_pl.loc[df_pl['date'] == playlist_date].index.values[0]
        except IndexError:
            raise ValueError(f'No monthly playlist found for date {str(playlist_date)}.')
        return pl_id

    def playlist_date(self, playlist_id:str) -> datetime.date:
        '''
        Get the date of a monthly playlist from its Spotify ID.

        Parameters
        ----------
        playlist_id : str
            The Spotify ID of the playlist.

        Returns
        -------
        datetime.date
            The date of the playlist.
        '''
        df_pl = self.read_monthly_playlists()
        try:
            pl_date = df_pl.loc[playlist_id, 'date']
        except KeyError:
            raise ValueError(f'No monthly playlist found for ID {playlist_id}.')
        return pl_date

    def get_playlist_cover_image(self, *args, playlist_id:str = None, playlist_date:Union[str, datetime.date] = None,
                                 overwrite:bool = False, errors:str = 'raise') -> np.ndarray:
        '''
        Get the cover image for a monthly playlist, either by its Spotify ID or date.
        The image is saved in the data/imgs directory as a .jpeg file, named
        cover_YYYY-MM.jpeg.

        Parameters
        ----------
        playlist_id : str, optional
            The Spotify ID of the playlist. If None, playlist_date must be provided.
        playlist_date : str or datetime.date, optional
            The date of the playlist. If a string, should be in the form YYYY-MM-DD.
            If None, playlist_id must be provided.
        overwrite : bool
            If True, will try to download the image even if it already exists. If False (default),
            will skip downloading if the file has already been saved and return the saved image.
        errors : {'raise', 'ignore'}
            If 'raise' (default), will raise an error if the playlist does not exist or has no cover image.
            If 'ignore', will return None in these cases.
        
        Returns
        -------
        np.ndarray
            The cover image as a numpy array, or None if no cover image exists.
        '''
        if args:
            if len(args) == 1:
                if isinstance(args[0], str):
                    if len(args[0]) == 10 and args[0][4] == '-' and args[0][7] == '-':
                        playlist_date = args[0]
                    else:
                        playlist_id = args[0]
                elif isinstance(args[0], datetime.date):
                    playlist_date = args[0]
                else:
                    raise ValueError(f'Unexpected type "{type(args[0])}" for date/ID input encountered. Try entering date as either a string (YYYY-MM-DD) or datetime.date, or ID as a string.')
            else:
                raise ValueError('Only one positional argument is accepted, either playlist_id or playlist_date.')


        # check inputs are valid
        if (playlist_id is None) and (playlist_date is None):
            raise ValueError('Must provide either playlist_id or playlist_date.')
        if (playlist_id is not None) and (playlist_date is not None):
            raise ValueError('Must provide only one of playlist_id or playlist_date.')
        
        # get ID from date
        if playlist_date is not None:
            try:
                playlist_id = self.playlist_id(playlist_date)
            except ValueError as e:
                if errors == 'ignore':
                    return None
                else:
                    raise e
            if isinstance(playlist_date, str):
                playlist_date = pd.to_datetime(playlist_date).date()
        elif playlist_id is not None:
            try:
                playlist_date = self.playlist_date(playlist_id)
            except ValueError as e:
                if errors == 'ignore':
                    return None
                else:
                    raise e
        
        # check if image already exists, and return if so
        save_path = os.path.join(self.data_dir, 'imgs', f'cover_{playlist_date.year}-{playlist_date.month:02d}.jpeg')
        if (not overwrite) and os.path.exists(save_path):
            return plt.imread(save_path)

        df_pl = self.read_monthly_playlists()

        # try to get the image url, and complain if it doesn't exist
        try:
            pl_im_url = df_pl.loc[playlist_id, 'cover_image_url']
        except KeyError:
            if errors == 'ignore':
                return None
            
            if playlist_date is not None:
                raise ValueError(f'No monthly playlist found for date {str(playlist_date)}.')
            else:
                raise ValueError(f'No monthly playlist found for ID {playlist_id}.')
        
        # playlist exists, but has no cover image
        if pd.isna(pl_im_url):
            if errors == 'ignore':
                return None
            
            if playlist_date is not None:
                raise ValueError(f'No cover image found for date {str(playlist_date)}.')
            else:
                raise ValueError(f'No cover image found for ID {playlist_id}.')


        r = requests.get(pl_im_url)
        im = plt.imread(BytesIO(r.content), format='jpeg')
        plt.imsave(save_path, im)
        return im
    
    def plot_playlist_covers(self, orientation:str = 'horizontal', 
                             scale:float = 1.0, wspace:float = 0.1, hspace:float = 0.1) -> Tuple[plt.Figure, np.ndarray]:
        '''
        Plot a collage of all monthly playlist cover images.

        Parameters
        ----------
        orientation : {'horizontal', 'vertical'}
            If 'horizontal' (default), the plot will have years as rows and months as columns.
            If 'vertical', the plot will have months as rows and years as columns.
        scale : float
            A scaling factor to apply to the size of each image in the plot. Default is 1.0.
            Applied to the figsize parameter of plt.subplots().
        wspace : float
            The amount of width reserved for space between subplots, expressed as a fraction
            of the average axis width. Default is 0.1.
        hspace : float
            The amount of height reserved for space between subplots, expressed as a fraction
            of the average axis height. Default is 0.1.
        
        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.
        np.ndarray
            The array of Axes objects in the plot.
        '''
        df_pl = self.read_monthly_playlists()
        n_years = df_pl['date'].apply(lambda x: x.year).nunique()
        min_year = df_pl['date'].apply(lambda x: x.year).min()
        n_months = df_pl['date'].apply(lambda x: x.month).nunique()  # should be 12 but this covers the special case of a new user with less than a years data

        if orientation == 'horizontal':
            n_rows, n_cols = n_years, n_months
        else:
            n_rows, n_cols = n_months, n_years
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*scale, n_rows*scale))

        for row_i, months_axs in enumerate(axes):
            for col_i, ax in enumerate(months_axs):
                if orientation == 'horizontal':
                    y, m = row_i, col_i
                else:
                    y, m = col_i, row_i
                date = f'{min_year+y}-{m+1:02d}-01'
                im = self.get_playlist_cover_image(playlist_date=date, errors='ignore')
                if im is not None:
                    ax.imshow(self._centre_square_crop(im))

                ax.axis('off')
        
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

        return fig, axes
    
    @staticmethod
    def _centre_square_crop(im):
        h, w, _ = im.shape
        if h == w:
            return im
        elif h > w:
            diff = h - w
            pad1 = diff // 2
            pad2 = diff - pad1
            return im[pad1:-pad2, :, :]
        else:
            diff = w - h
            pad1 = diff // 2
            pad2 = diff - pad1
            return im[:, pad1:-pad2, :]

    def get_tracks(self, backoff_time: float = 5.0, progress_bar: bool = True,
                   to_csv: bool = True, date_csv: bool = False) -> pd.DataFrame:
        '''
        Get all tracks in all monthly playlists.

        Parameters
        ----------
        backoff_time : float
            Time in seconds to wait between API calls to avoid rate limiting.
            Default is 5.0 seconds. Only applies for playlists that require more than 1 call (i.e., playlists with more than 100 tracks).
        progress_bar : bool
            If True (default), will display a progress bar while fetching data. Note rate estimates
            may be inaccurate if backoff_time is large.
        to_csv : bool
            If True, will save the resulting pd.DataFrame as a CSV in the data directory.
        date_csv : bool
            If True, the saved CSV will have the date of data collection in the filename. Set to
            True to avoid overwriting previous files.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing all tracks in all monthly playlists, with one row per artist per track
        '''

        df_mpls = self.read_monthly_playlists(download_if_required=True)

        track_data = dict(
            track_name = [],
            track_artist = [],
            track_date_added = [],
            playlist_id = [],
            playlist_name = [],
            track_index = [],  # 0-indexed position in the playlist
            track_artist_index = [],  # 0-indexed order of artist on the track
            track_artist_spid = [],
            track_album = [],
            track_release_date = [],
            track_release_date_precision = [],
            track_duration = [],
            track_popularity = [],  # note this will change over time, as it depends on number of listens and how recent those listens are
            track_external_ids = [],  # probably not useful
            track_spid = []
        )
        artist_ids = set()


        # get track info for all monthly playlists
        for id, playlist_data in tqdm(df_mpls.iterrows(), total=len(df_mpls), disable=not progress_bar):
            n_calls = np.ceil(playlist_data['n_tracks'] / 100)

            for call in range(int(n_calls)):
                if call > 0:
                    time.sleep(backoff_time)  # shhh little spotify api client. do not complain. it will all be ok. just rest.

                pl_mth = self.spotify_client.playlist_items(id, limit=100, offset=call*100)

                # iterate through tracks
                for track_i, track in enumerate(pl_mth['items']):

                    # iterate through each track's artists (can have multiple)
                    for artist_i, artist in enumerate(track['track']['artists']):
                        # append all data to lists:
                        # playlist info
                        track_data['playlist_id'].append(id)
                        track_data['playlist_name'].append(df_mpls.loc[id, 'name'])
                        track_data['track_index'].append(track_i + call*100)

                        # track name, album, artists
                        track_data['track_date_added'].append(track['added_at'])
                        track_data['track_name'].append(track['track']['name'])
                        track_data['track_artist'].append(artist['name'])
                        track_data['track_artist_index'].append(artist_i)
                        track_data['track_artist_spid'].append(artist['id'])
                        track_data['track_album'].append(track['track']['album']['name'])

                        # extract artist id for bulk requests later
                        artist_ids.add(artist['id'])

                        # track info
                        track_data['track_release_date'].append(track['track']['album']['release_date'])
                        track_data['track_release_date_precision'].append(track['track']['album']['release_date_precision'])
                        track_data['track_popularity'].append(track['track']['popularity'])
                        track_data['track_duration'].append(track['track']['duration_ms'])
                        track_data['track_external_ids'].append(track['track']['external_ids'])
                        track_data['track_spid'].append(track['track']['id'])

        artist_genres = dict()
        artist_popularities = dict()

        # get artist data in blocks of 50
        artist_ids_list = list(artist_ids)
        for i in tqdm(range(0, len(artist_ids), 50)):
            artist_block = artist_ids_list[i:i+50]
            artists_expanded = self.spotify_client.artists(artist_block)
            time.sleep(0.1)  # catnap
            for artist in artists_expanded['artists']:
                artist_genres[artist['id']] = artist['genres']
                artist_popularities[artist['id']] = artist['popularity']


        df_tracks = pd.DataFrame(track_data)
        df_tracks['track_date_added'] = pd.to_datetime(df_tracks['track_date_added'])
        df_tracks['track_artist_genres'] = df_tracks['track_artist_spid'].map(lambda name: artist_genres.get(name, []))
        df_tracks['track_artist_popularity'] = df_tracks['track_artist_spid'].map(lambda name: artist_popularities.get(name, np.nan))

        def format_release_date(date):
            if date == '0000':
                return pd.NA
            if len(date) == 4:
                return date + '-01-01'
            elif len(date) == 7:
                return date + '-01'
            else:
                return date
        df_tracks['track_release_date'] = pd.to_datetime(df_tracks['track_release_date'].apply(format_release_date))

        if to_csv:
            if date_csv:
                filename = f'tracks_{str(datetime.datetime.now().date())}.csv'
            else:
                filename = 'tracks.csv'
            df_tracks.to_csv(
                os.path.join(self.data_dir, filename),
                index=False
            )
        return df_tracks

    def read_tracks(self, date = None, download_if_required = False) -> pd.DataFrame:
        '''
        Read a saved DataFrame of all tracks in all monthly playlists.

        Parameters
        ----------
        date : str or datetime.date
            The date the data was collected. If None, will attempt to read the undated file,
            which is assumed to be the most recent. If the undated file does not exist,
            will read the most recent dated file.
            If the date is specified as a string, it should be given in the form YYYY-MM-DD.

        download_if_required : bool
            If True, will download the data using `get_tracks()` if the requested
            file does not exist.
            If False (default), will raise a FileNotFoundError if the requested file does not exist.

        Returns
        -------
        pandas.DataFrame
        '''
        
        if date is None:
            file = 'tracks.csv'
        elif isinstance(date, datetime.date):
            file = f'tracks_{str(date.date())}.csv'
        elif isinstance(date, str):
            file = f'tracks_{date}.csv'
        else:
            raise ValueError(f'Unexpected type "{type(date)}" for date input encountered. Try entering date as either a string (YYYY-MM-DD) or datetime.date.')
        
        if not os.path.exists(os.path.join(self.data_dir, file)):
            if download_if_required:
                if date is None:
                    df = self.get_tracks(to_csv=True)
                else:
                    raise FileNotFoundError(f'File {file} does not exist in data directory {self.data_dir}. Cannot download data for a specific date.')
            else:
                raise FileNotFoundError(f'File {file} does not exist in data directory {self.data_dir}. Set download_if_required=True to download the data.')
        else:
            df = pd.read_csv(
                os.path.join(self.data_dir, file), 
            )
            df['track_date_added'] = pd.to_datetime(df['track_date_added'])
            df['track_release_date'] = pd.to_datetime(df['track_release_date'])
        
        return df

    @property
    def df_tracks(self) -> pd.DataFrame:
        '''
        Shortcut to read the most recent tracks DataFrame.

        Returns
        -------
        pandas.DataFrame
            The most recently saved DataFrame of all tracks in all monthly playlists.
        '''
        return self.read_tracks(download_if_required=True)
    
    @property
    def df_pl(self) -> pd.DataFrame:
        '''
        Shortcut to read the most recent monthly playlists DataFrame.

        Returns
        -------
        pandas.DataFrame
            The most recently saved DataFrame of all monthly playlists.
        '''
        return self.read_monthly_playlists(download_if_required=True)

    @property
    def ids(self) -> list:
        '''
        Get a list of all monthly playlist IDs.

        Returns
        -------
        list
            A list of all monthly playlist Spotify IDs.
        '''
        df_pl = self.read_monthly_playlists(download_if_required=False)
        return df_pl.index.tolist()
    
    @property
    def ids_to_dates(self) -> dict:
        '''
        Dictionary mapping all monthly playlist IDs to their dates.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist IDs to their dates.
        '''
        df_pl = self.read_monthly_playlists(download_if_required=False)
        return df_pl['date'].to_dict()

    @property
    def ids_to_names(self) -> dict:
        '''
        Dictionary mapping all monthly playlist IDs to their names.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist IDs to their names.
        '''
        df_pl = self.read_monthly_playlists(download_if_required=False)
        return df_pl['name'].to_dict()
    
    @property
    def names_to_ids(self) -> dict:
        '''
        Dictionary mapping all monthly playlist names to their IDs.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist names to their IDs.
        '''
        df_pl = self.read_monthly_playlists(download_if_required=False)
        return df_pl.reset_index().set_index('name')['id'].to_dict()
    
    @property
    def dates_to_ids(self) -> dict:
        '''
        Dictionary mapping all monthly playlist dates to their IDs.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist dates to their IDs.
        '''
        df_pl = self.read_monthly_playlists(download_if_required=False)
        return df_pl.reset_index().set_index('date')['id'].to_dict()
    
    @property
    def genres(self) -> np.ndarray:
        '''
        Set of genres present in all monthly playlists

        Returns
        -------
        set
            A set of all genres present in all monthly playlists.
        '''
        return np.array(list(set(genre for genres in self.df_tracks['track_artist_genres'] for genre in eval(genres))))

    @property
    def supergenres(self) -> np.ndarray:
        return np.array(list(self.supergenre_lists.keys()))

    @property
    def artists(self) -> np.ndarray:
        '''
        Set of artists present in all monthly playlists

        Returns
        -------
        np.ndarray
            An array of all artists present in all monthly playlists.
        '''
        return self.df_tracks['track_artist'].unique()

    @property
    def artist_counts(self) -> pd.Series:
        '''
        Series of artist name: number of times that artist appears in `df_tracks`.

        Returns
        -------
        pd.Series
        '''
        return self.df_tracks['track_artist'].value_counts()

    @property
    def dates(self) -> np.ndarray:
        '''
        Dates of all playlists present in the downloaded playlists.csv file

        Returns
        -------
        np.ndarray
            An array of the dates (months) the monthly playlist data covers
        '''
        return self.df_pl['date'].unique()
    
    @property
    def artist_timeseries(self) -> pd.DataFrame:
        '''
        A dataframe with index of playlist dates, and columns of artists, with
        values showing how many tracks by/featuring that artist are present in the given months playlist.

        Returns
        -------
        pd.DataFrame
        '''
        df = self.df_tracks.groupby('playlist_id')['track_artist'].value_counts().reset_index().pivot(
            columns = 'track_artist', index = 'playlist_id',values = 'count'
        ).fillna(0)

        df.index = df.index.map(self.ids_to_dates).set_names('playlist_date')
        return df.sort_index()
    
    def tracks_with_artist_genre(self, genre, comparison_type = 'exact', *, supergenre = False):
        '''
        Get all tracks in the monthly playlists that have a given artist genre. The artist genre can
        be a genre (default) or, if `supergenre = True`, a supergenre as specified in `supergenre_lists`.

        Parameters
        ----------
        genre : str
            The genre to search for. Should be one of the available genres (see `genres`) or 
            a supergenre (`supergenres`).
        comparison_type : {'exact', 'like'}
            How to compare the given genre string to the genre list. Defaults to 'exact'. Ignored when `supergenre = True`
            - 'exact': Only return tracks with an exact match of the given genre.
            - 'like': Return any track where `genre` is a substring of one or more of the track's artist genres.
                      Useful for finding similar genres (e.g. 'hip-hop' and 'old school hip-hop')
        supergenre : bool, optional
            Whether or not the given genre should be interpreted as a supergenre. Defaults to False.
        '''
        if supergenre:
            find_fn = lambda x: genre.upper() in [self.supergenre_map[i] for i in x]
        else:
            if comparison_type == 'exact':
                find_fn = lambda x: genre.lower() in x
            elif comparison_type == 'like':
                find_fn = lambda x: genre.lower() in ','.join(x)

        return self.df_tracks.loc[self.df_tracks['track_artist_genres'].apply(find_fn)]
        
    def n_tracks_with_artist_genre(self, genre, comparison_type = 'exact', supergenre = False):
        '''
        The number of tracks in the monthly playlists that have a given artist genre. The artist genre can
        be a genre (default) or, if `supergenre = True`, a supergenre as specified in `supergenre_lists`.

        Parameters
        ----------
        genre : str
            The genre to search for. Should be one of the available genres (see `genres`) or 
            a supergenre (`supergenres`).
        comparison_type : {'exact', 'like'}
            How to compare the given genre string to the genre list. Defaults to 'exact'. Ignored when `supergenre = True`
            - 'exact': Only return tracks with an exact match of the given genre.
            - 'like': Return any track where `genre` is a substring of one or more of the track's artist genres.
                      Useful for finding similar genres (e.g. 'hip-hop' and 'old school hip-hop')
        supergenre : bool, optional
            Whether or not the given genre should be interpreted as a supergenre. Defaults to False.
        '''
        return self.tracks_with_artist_genre(genre, comparison_type, supergenre = supergenre).groupby('track_index').ngroups
    
    @property
    def genre_track_counts(self):
        '''
        Get the number of tracks of each genre.

        Returns
        -------
        pd.Series
            Series of track counts indexed by genre names
        '''
        return pd.Series({genre: self.n_tracks_with_artist_genre(genre, 'exact') for genre in self.genres}).sort_values(ascending=False)
    
    @property
    def supergenre_track_counts(self):
        '''
        Get the number of tracks of each supergenre, as defined in `supergenre_lists`.

        Returns
        -------
        pd.Series
            Series of track counts indexed by supergenre names
        '''
        return pd.Series({genre: self.n_tracks_with_artist_genre(genre, 'exact', supergenre=True) for genre in self.supergenres}).sort_values(ascending=False)

    def plot_supergenres(self,
                    highlight = 'funk', 
                    xaxis_pad = pd.Timedelta('30D')):
        

        ################################################## FLATTEN ARTIST GENRES ##################################################

        track_cols = [  # excluding columns that specify artists in the aggregation
            'track_names',  
            'track_date_added',
            'playlist_id',
            'playlist_name',
            'track_index',
            'track_album',
            'track_release_date',
            'track_release_date_precision',
            'track_duration',
            'track_popularities',
            'track_external_ids',
            'track_spid'
        ]

        df_tracks = self.df_tracks
        df_tracks['track_artist_genres'] = df_tracks['track_artist_genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)

        df_tracks = df_tracks.groupby(track_cols).agg({
            'track_artist_genres' : lambda x: list(set(x.sum())),
            'track_artist': lambda x: x.to_list(),
            'track_artist_spid' : lambda x: x.to_list(),
            'track_artist_popularity' : lambda x: x.to_list(),
        }).sort_index(
            level = 'track_date_added', 
            ascending=False
        ).reset_index()


        ###################################################### ONE HOT ENCODE ######################################################    
        genres = sorted(list(self.supergenre_lists.keys()))
        agg_fn = lambda x: genre in [self.supergenre_map[i] for i in x]

        cols = []
        for genre in genres:
            ser = df_tracks['track_artist_genres'].apply(agg_fn)
            ser.name = genre
            cols.append(ser)
            
        df_tracks = pd.concat([df_tracks, pd.concat(cols, axis = 1)], axis = 1)


        ######################################################## SUM BY MONTH ########################################################    

        # genre sums in each playlist
        df_genres = df_tracks.groupby('playlist_id')[genres].agg(lambda x: sum(x))  # /len(x))  # no percentages!
        df_genres.index = df_genres.index.map(self.ids_to_dates).set_names('playlist_date')
        df_genres.sort_index(inplace=True)

        ############################################### CALCULATE RANKS IN EACH MONTH ###############################################    
        df_rank = df_genres.replace(0, pd.NA).rank(
            ascending=False, 
            method = 'first', 
            na_option='keep',
            axis = 1
        ).sort_values(
            axis = 1, 
            by = df_genres.index.values[1],
            ascending=True
        )
        
        ########################################################### PLOT ###########################################################    
        fig, ax = plt.subplots(figsize = (20,5))
        cmap = plt.get_cmap('tab20')

        for i, col in enumerate(df_rank):
            if col == highlight.upper():
                ax.plot(
                    df_rank[col], 
                    color = 'r', 
                    label = col,
                    linestyle = 'None', 
                    marker = 'o',
                )
            else:
                ax.plot(
                    df_rank[col], 
                    color = cmap(i), 
                    label = col, 
                    alpha = 0.7,
                    linestyle = 'None', 
                    marker = 'x'
                )

        ax.set_yticks(range(1,19))


        ax.set_xlim(df_genres.index.min() - xaxis_pad, df_genres.index.max() + xaxis_pad)
        ax.invert_yaxis()
        ax.legend(bbox_to_anchor = (1, 0.95), title = 'Genres')
        #ax.set_title('Genre ranking timeseries')
        ax.set_xlabel('Month')
        ax.set_ylabel('Rank')
        fig.set_constrained_layout(True)

        # set axis box off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.xaxis.set_minor_locator(m_dates.MonthLocator())

        plt.show()    

class LoggingSpotifyClient(spotipy.Spotify):
    '''
    Wrapper class to keep track of Spotify API calls.
    '''

    RATE_LIMIT_PER_30S = 90
    log_filepath = os.path.join(LOG_DIR, 'spotify_api_calls.csv')

    def __init__(self, spotify_client = None, *args, **kwargs):
        '''
        Create a new `LoggingSpotifyClient` instance, optionally wrapping an existing
        `spotipy.Spotify` instance. See `spotipy.Spotify` documentation for
        accepted args and kwargs.

        Parameters
        ----------
        spotify_client : spotipy.Spotify, optional
            The spotify client to wrap the logger around. If None, a new instance
            will be created using the provided args and kwargs.
        '''
        self.log_action('Initialized new LoggingSpotifyClient instance.')

        if spotify_client is not None:
            if not isinstance(spotify_client, spotipy.Spotify):
                raise ValueError(f'Expected spotify_client to be of type spotipy.Spotify, got {type(spotify_client)}.')
            self.__dict__.update(spotify_client.__dict__)
            self.__class__ = type('LoggingSpotifyClient', (self.__class__, spotify_client.__class__), {})
        else:
            super().__init__(*args, **kwargs)
        
        self.clean_log()  # clear old log entries

    def _internal_call(self, method, url, payload, params):
        self.log_call(method, check_rate=True)
        return super()._internal_call(method, url, payload, params)
    
    def log_call(self, call_name, check_rate:bool = True, **check_rate_kwargs):
        '''
        Log the call time and type to the log file. 

        Parameters
        ----------
        call_name : str
            The name of the API call being made.
        check_rate : bool
            If True (default), will check the current API rate and sleep if necessary
            to avoid exceeding the rate limit.
        **check_rate_kwargs
            Optional keyword arguments to pass to `check_api_rate()`.
        '''
        logger.debug(call_name)
        if check_rate:
            self.check_api_rate(**check_rate_kwargs)

    @property
    def df_log(self) -> pd.DataFrame:
        '''
        Read the API call log file into a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The log file as a DataFrame.
        '''
        try:
            df_log = pd.read_csv(self.log_filepath, parse_dates=['time'])
            return df_log
        except pd.errors.EmptyDataError:
            # case where log file is empty
            with open(self.log_filepath, 'w') as f:
                f.write('time,message\n')
            return pd.DataFrame(columns=['time', 'message'])

    @property
    def api_rate(self) -> int:
        '''
        Check how many API calls have been made in the last 30 second window.

        Returns
        -------
        int
            The number of API calls made in the last 30 seconds.
        '''
        window_s = 30
        df_log = self.df_log
        now = pd.Timestamp.now()
        window_start = now - pd.Timedelta(seconds=window_s)
        return df_log[df_log['time'] >= window_start].shape[0]

    def check_api_rate(self, limit_proportion:float = 0.8,
                       sleep_time:float = 5.0, warn:bool = True):
        '''
        Determine if the current API rate is below `limit_proportion` of the approximate allowed threshold (180 calls/min). If
        over the threshold, sleep for `sleep_time` seconds and check again, repeating until the
        rate is below the threshold.

        Parameters
        ----------
        limit_proportion : float
            The proportion of the rate limit to check against (default is 0.8 for 80%).
        sleep_time : float
            The number of seconds to sleep for if the API rate is over the limit.
        '''
        near_limit = self.api_rate > (self.RATE_LIMIT_PER_30S * limit_proportion)  # default is 80% of limit

        if near_limit and sleep_time > 0:
            if warn:
                logger.warning(f'Approaching API {limit_proportion} of rate limit ({self.api_rate} > {self.RATE_LIMIT_PER_30S * limit_proportion} = {self.RATE_LIMIT_PER_30S} * {limit_proportion}), sleeping for {sleep_time} seconds.')
            time.sleep(sleep_time)
            self.check_api_rate(sleep_time=sleep_time, # keep checking until we're under the limit
                                limit_proportion=limit_proportion, 
                                warn=False)  # don't keep warning I don't want to go crazy
            
    def clean_log(self):
        '''
        Delete all previous log entries that are not within the last 30 seconds.
        '''
        window_s = 30
        df_log = self.df_log
        now = pd.Timestamp.now()
        window_start = now - pd.Timedelta(seconds=window_s)
        df_log = df_log[df_log['time'] >= window_start]
        df_log.to_csv(self.log_filepath, index=False)
        self.log_action('Cleaned old log entries.')
    
    def log_action(self, message:str):
        '''
        Log a non-API action to the log file.

        Parameters
        ----------
        message : str
            The message to log.
        '''
        logger.info(message)

class MonthlyPlaylist:
    '''
    Object encompassing all data for a monthly playlist
    '''
    playlists_filepath = os.path.join(DATA_DIR, 'playlists.csv')
    tracks_filepath = os.path.join(DATA_DIR, 'tracks.csv')

    def __init__(self, identifier = None, /, sp_id=None, date=None, name=None):
        '''
        Create a MonthlyPlaylist object, identified by either its Spotify ID, date, or name.
        This object contains all data about the playlist, including track data, cover image, and
        relevant metadata.

        Parameters
        ----------
        identifier : str or datetime.date or datetime.datetime, optional
            A single identifier for the playlist, which can be either its Spotify ID (str),
            date (str, datetime.date or datetime.datetime), or name (str). If provided, this takes
            precedence over the other parameters.
        sp_id : str, optional
            The Spotify ID of the playlist. Must be 22 characters long. Ignored if `identifier` is provided.
        date : str or datetime.date or datetime.datetime, optional
            The date associated with the playlist. If a string is provided, it must be in the
            format 'YYYY-MM-01'. Ignored if `identifier` is provided.
        name : str, optional
            The name of the playlist. Ignored if `identifier` is provided.

        Raises
        ------
        FileNotFoundError
            If the required data files (playlists.csv and tracks.csv) are not found in the
            expected data directory.
        ValueError
            If the provided identifier cannot be identified as a Spotify ID, date, or name.
        '''
        if not self._playlists_file_exists or not self._tracks_file_exists:
            missing_files = ' '.join([f"{self.playlists_filepath if not self._playlists_file_exists else ''}", 
                                      f"{self.tracks_filepath if not self._tracks_file_exists else ''}"])
            raise FileNotFoundError(f"Tried initialising MonthlyPlaylist object, but missing: {missing_files}. Playlist data must be downloaded to initialise a playlist object.")

        if identifier:
            id_type = self._identify_identifier(identifier)
            if id_type == 'sp_id':
                sp_id = identifier

            elif id_type == 'date':
                date = identifier

            elif id_type == 'name':
                name = identifier
            else:
                raise ValueError(f'Unidentified identifier type "{identifier}" encountered.')
        
        df_mpls = MonthlyPlaylistHandler().read_monthly_playlists()
        if sp_id:
            data = df_mpls.loc[sp_id]
        elif date:
            data = df_mpls.loc[df_mpls['date'] == pd.to_datetime(date).date()].iloc[0]
        elif name:
            data = df_mpls.loc[df_mpls['name'] == name].iloc[0]

        for item in data.index:
            setattr(self, item, data[item])
        self.sp_id = data.name

    def _identify_identifier(self, identifier):
        '''Identify whether a given identifier is a spotfy id, date, or playlist name.'''
        
        if isinstance(identifier, (datetime.date, datetime.datetime)):
            return 'date'
        
        elif isinstance(identifier, str):
            if identifier[:2] == '20' and identifier.count('-') == 2 and identifier[-2:] == '01':
                return 'date'
            elif len(identifier) == 22 and ' ' not in identifier and '-' not in identifier:
                return 'sp_id'
            else:
                return 'name'
            
    @property
    def df_tracks(self):
        df_tracks = MonthlyPlaylistHandler().read_tracks()
        df_tracks['track_artist_genres'] = df_tracks['track_artist_genres'].apply(lambda x: eval(x))
        return df_tracks.loc[df_tracks['playlist_id'] == self.sp_id]

    def __getitem__(self, key):
        return self.df_tracks[key]
    
    @property
    def loc(self):
        return self.df_tracks.loc
    
    @property
    def genres(self):
        return set(self['track_artist_genres'].sum())
    
    @property
    def artists(self):
        return set(self['track_artist'].unique())
    
    @property
    def artist_track_counts(self):
        return self['track_artist'].value_counts()
    
    def tracks_with_artist_genre(self, genre, comparison_type = 'exact'):
        if comparison_type == 'exact':
            return self.loc[self['track_artist_genres'].apply(lambda x: genre in x)]
        elif comparison_type == 'like':
            return self.loc[self['track_artist_genres'].apply(lambda x: genre in ','.join(x))]
    
    def tracks_with_artist(self, artist_name):
        return self.loc[self['track_artist'].str.lower() == artist_name.lower()]

    def n_tracks_with_artist_genre(self, genre, comparison_type = 'exact'):
        return self.tracks_with_artist_genre(genre, comparison_type).groupby('track_index').ngroups

    def n_tracks_with_artist(self, artist_name):
        return self.tracks_with_artist(artist_name).shape[0]

    def genre_track_counts(self, genres = None):
        '''
        Get the number of tracks
        '''
        if genres is None:
            genres = self.genres
        return {genre: self.n_tracks_with_artist_genre(genre, 'exact') for genre in genres}

    @property
    def cover_img_filepath(self):
        return os.path.join(IMG_DIR, f"cover_{self.date.year}-{self.date.month:02d}.jpeg")
    
    @property
    def cover_img(self):
        if self._cover_img_exists:
            return plt.imread(self.cover_img_filepath)
        else:
            raise FileNotFoundError(f"Cover image file not found at expected location: {self.cover_img_filepath}")
    
    def plot_cover_img(self, ax = None, **kwargs):
        '''
        Plot the cover image of the playlist using plt.imshow.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot the image on. If None (default), will be plot on its own axes using plt.imshow()
        kwargs : dict, optional
            Optional keyword arguments to pass to plt.imshow/ax.imshow

        Returns
        -------
        ax
            The created or modified axes object
        '''
        if ax:
            ax.imshow(self.cover_img, **kwargs)
        else:
            ax = plt.imshow(self.cover_img, **kwargs)
        
        return ax
    
    def __repr__(self):
        return f"""MonthlyPlaylist: {self.name}"""

    @property
    def _playlists_file_exists(self):
        return os.path.exists(self.playlists_filepath)
    
    @property
    def _tracks_file_exists(self):
        return os.path.exists(self.tracks_filepath)

    @property
    def _cover_img_exists(self):
        return os.path.exists(self.cover_img_filepath)
