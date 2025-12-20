import os
import spotipy
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
from .genres import supergenre_lists, supergenre_map
import duckdb

# warning begone 🪄
pd.set_option('future.no_silent_downcasting', True)

# default max number of artists to be stored in tracks.csv. Will be expanded if more artists are found on a track.
N_ARTISTS_MAX = 10

# ensure data directory is set up as expected
DATA_DIR = os.path.abspath('data')
IMG_DIR = os.path.join(DATA_DIR, 'imgs')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
MPL_DIR = os.path.join(DATA_DIR, 'mpls')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(MPL_DIR):
    os.makedirs(MPL_DIR)
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

class MonthlyPlaylistHandler:
    '''
    Handler for monthly playlist data. Contains a spotipy Spotify instance,
    handling authentication.
    '''
    
    data_dir = DATA_DIR
    img_dir = IMG_DIR
    mpl_dir = MPL_DIR

    n_artists_max = N_ARTISTS_MAX

    supergenre_lists = supergenre_lists
    supergenre_map = supergenre_map

    playlists_file = os.path.join(data_dir, 'playlists.csv')
    tracks_file = os.path.join(data_dir, 'tracks.csv')
    artists_file = os.path.join(data_dir, 'artists.csv')
    artist_genres_file = os.path.join(data_dir, 'artist_genres.csv')
    
    data_files = [playlists_file, tracks_file, artists_file, artist_genres_file]

    year_styles = {
        "2018":"2018",
        "2019":"2019",
        "2020":"2020",
        "2021":"2021",
        "2022":" 22",
        "2023":" 23",
        "2024":"-24",
        "2025":"2025"
    }
    
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
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #################################### DOWNLOADS ####################################
    def download_playlist_metadata(self):
        '''
        Download a csv containing monthly playlists metadata. See `MonthlyPlaylistHandler.playlist_file` for output.
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
        df['date'] = pd.to_datetime(df['date'], format = r'%m-%Y').dt.date

        # extract image url
        df['cover_image_url'] = df['images'].apply(lambda x: x[0]['url'] if not pd.isna(x[0]) else x)

        # rename href to url
        df = df.rename(columns={'href':'url'})

        # only keep useful columns
        df = df[['id', 'date', 'name', 'description', 'n_tracks', 'url', 'cover_image_url', 'snapshot_id']].set_index('id')

        # save
        df.to_csv(self.playlists_file, index = True)

    def download_playlist_cover_image(self, identifier:Union[str, datetime.date],
                                      overwrite:bool = True, errors:str = 'raise'):
        '''
        Download the cover image for a monthly playlist, either by its Spotify ID or date.
        The image is saved in the data/imgs directory as a .jpeg file, named
        cover_YYYY_MM.jpeg.

        Parameters
        ----------
        identifier : str or datetime.date
            The identifier of the playlist - either a spotify playlist ID, date, or playlist name.
        overwrite : bool
            If True (default), will try to download the image even if it already exists. If False,
            will skip downloading if the file has already been saved and return the saved image.
        errors : {'raise', 'ignore'}
            If 'raise' (default), will raise an error if the playlist does not exist or has no cover image.
            If 'ignore', will raise no error and return early.
        
        Returns
        -------
        np.ndarray
            The cover image as a numpy array, or None if no cover image exists.
        '''

        playlist_date = self.convert_playlist_identifier(identifier, 'date')
        playlist_id = self.convert_playlist_identifier(identifier, 'id')
        
        # check if image already exists, and return if so
        save_path = os.path.join(self.data_dir, 'imgs', f'cover_{playlist_date.year}_{playlist_date.month:02d}.jpeg')
        if (not overwrite) and os.path.exists(save_path):
            return plt.imread(save_path)

        # try to get the image url, and complain if it doesn't exist
        try:
            pl_im_url = self.df_playlists.loc[playlist_id, 'cover_image_url']
        except KeyError:
            if errors == 'ignore':
                return
            
            elif playlist_date is not None:
                raise ValueError(f'No monthly playlist found for date {str(playlist_date)}.')
            else:
                raise ValueError(f'No monthly playlist found for ID {playlist_id}.')
        
        # playlist exists, but has no cover image
        if pd.isna(pl_im_url):
            if errors == 'ignore':
                return
            
            elif playlist_date is not None:
                raise ValueError(f'No cover image found for date {str(playlist_date)}.')
            else:
                raise ValueError(f'No cover image found for ID {playlist_id}.')


        r = requests.get(pl_im_url)
        im = plt.imread(BytesIO(r.content), format='jpeg')
        plt.imsave(save_path, im)
    
    def download_playlist_cover_images(self, overwrite:bool = True, errors:str = 'raise'):
        '''
        Download all cover images of monthly playlists. 
        The images are saved in the data/imgs directory as .jpeg files, named cover_YYYY_MM.jpeg.

        Parameters
        ----------
        overwrite : bool
            If True (default), will try to download the image even if it already exists. If False,
            will skip downloading if the file has already been saved and return the saved image.
        errors : {'raise', 'ignore'}
            If 'raise' (default), will raise an error if the playlist does not exist or has no cover image.
            If 'ignore', will raise no error and return early.
        '''
        for playlist_id in self.playlist_ids:
            self.download_playlist_cover_image(
                identifier=playlist_id,
                overwrite=overwrite, 
                errors = errors
            )

    def download_playlist_contents(self, backoff_time: float = 0.0, progress_bar: bool = False):
        '''
        Get all tracks in all monthly playlists, 

        Parameters
        ----------
        backoff_time : float
            Time in seconds to sleep after API calls to avoid rate limiting.
            Default is 0.0 seconds. Only applies for playlists that require more than 1 call (i.e., playlists with more than 100 tracks).
        progress_bar : bool
            If True (default), will display a progress bar while fetching data. Note rate estimates
            may be inaccurate if backoff_time is large.
        to_csv : bool
            If True, will save the resulting pd.DataFrame as a CSV in the data directory.
        date_csv : bool
            If True, the saved CSV will have the date of data collection in the filename. Set to
            True to avoid overwriting previous files.
        '''

        df_mpls = self.df_playlists

        # to hold all tracks
        track_data = dict(
            id = [],
            name = [],
            album = [],
            album_id = [],
            duration = [],
            release_date = [],
            release_date_precision = [],
            popularity = [],  # note this will change over time, as it depends on number of listens and how recent those listens are
            external_ids = [],  # probably not useful
        )
        # add cols for artist_1 to artist_n_artists_max. may need to config n_artists_max if there's a track with more than 10 artists.
        for n in range(self.n_artists_max):
            track_data[f'artist_{n+1}'] = []
        artist_ids = set()


        # get track info for all monthly playlists
        for playlist_id, playlist_metadata in tqdm(df_mpls.iterrows(), total=len(df_mpls), disable=not progress_bar, desc = 'Collecting track data'):
            playlist_data = dict(
                id = [],  # 0-indexed track positional
                track = [],
                date_added = []
            ) 

            for call in range(int(np.ceil(playlist_metadata['n_tracks'] / 100))):  # calls made in blocks of up to 100
                if call > 0:
                    time.sleep(backoff_time)  # shhh little spotify api client. do not complain. it will all be ok. just rest.

                pl_mth = self.spotify_client.playlist_items(playlist_id, limit=100, offset=call*100)

                # iterate through tracks
                for track_i, track in enumerate(pl_mth['items']):
                    n_artists = len(track['track']['artists'])
                    if n_artists > self.n_artists_max:
                        self.n_artists_max += 1
                        track_data[f'artist_{self.n_artists_max+1}'] = [pd.NA for item in track_data['id']]
                        logger.warning(f'Track "{track["track"]["name"]}" has more than {self.n_artists_max} artists - modify N_ARTISTS_MAX in source?\nAlbum: {track["track"]["album"]["name"]}, playlist: {df_mpls.loc[id, "name"]}')

                    track_id = track['track']['id']

                    # collect playlist_DATE.csv data - indexed by track position in playlist
                    playlist_data['id'].append(track_i)
                    playlist_data['track'].append(track_id)
                    playlist_data['date_added'].append(track['added_at'])

                    # collect tracks.csv data if we haven't already
                    if track_id not in track_data['id']:
                        track_data['id'].append(track_id)
                        track_data['name'].append(track['track']['name'])
                        track_data['album'].append(track['track']['album']['name'])
                        track_data['album_id'].append(track['track']['album']['id'])
                        track_data['duration'].append(track['track']['duration_ms'])
                        track_data['release_date'].append(track['track']['album']['release_date'])
                        track_data['release_date_precision'].append(track['track']['album']['release_date_precision'])
                        track_data['popularity'].append(track['track']['popularity'])
                        track_data['external_ids'].append(track['track']['external_ids'])

                        # collect artist info
                        for i in range(self.n_artists_max):
                            if i < n_artists:
                                artist_id = track['track']['artists'][i]['id']
                                artist_ids.add(artist_id)
                                track_data[f'artist_{i+1}'].append(artist_id)
                            else:
                                track_data[f'artist_{i+1}'].append(pd.NA)

            df_playlist = pd.DataFrame(playlist_data).set_index('id')
            df_playlist['date_added'] = pd.to_datetime(df_playlist['date_added'])
            df_playlist.to_csv(
                self.playlist_file(playlist_id),
                index = True
            )

        artist_data = dict(
            id = [],
            name = [],
            popularity = [],
            genres = []
        )
        genres = set()

        # get artist data in blocks of 50
        artist_ids_list = list(artist_ids)
        for i in tqdm(range(0, len(artist_ids), 50), disable=not progress_bar, desc = 'Collecting artist data'):
            artist_block = artist_ids_list[i:i+50]
            artists_expanded = self.spotify_client.artists(artist_block)
            time.sleep(backoff_time)

            for artist in artists_expanded['artists']:
                for genre in artist['genres']:
                    genres.add(genre)
                artist_data['genres'].append(artist['genres'])
                artist_data['id'].append(artist['id'])
                artist_data['name'].append(artist['name'])
                artist_data['popularity'].append(artist['popularity'])
        
        df_artists = pd.DataFrame(artist_data).set_index('id')
        
        # save metadata
        df_artists.drop(columns = ['genres']).to_csv(
            self.artists_file,
            index = True
        )
        
        genre_cols = []
        for genre in genres:
            genre_col = df_artists.loc[:, 'genres'].apply(lambda x: genre in x)
            genre_col.name = genre
            genre_cols.append(genre_col)
        
        # melt into three columns: id, genre, present. long and thin table.
        df_artists = pd.concat(
            [df_artists.drop(columns = ['name', 'genres', 'popularity']), pd.concat(genre_cols, axis = 1)], 
            axis = 1
        ).melt(ignore_index=False, var_name='genre', value_name='present')
        df_artists = df_artists.loc[df_artists['present'], 'genre']  # only keep genres that are present, and remove the present column

        df_tracks = pd.DataFrame(track_data).set_index('id')

        def format_release_date(date):
            if date == '0000':
                return pd.NA
            if len(date) == 4:
                return date + '-01-01'
            elif len(date) == 7:
                return date + '-01'
            else:
                return date
        df_tracks['release_date'] = pd.to_datetime(df_tracks['release_date'].apply(format_release_date), format = r'%Y-%m-%d')

        df_tracks.to_csv(
            self.tracks_file,
            index=True
        )
        df_artists.to_csv(
            self.artist_genres_file,
            index=True
        )

    def download(self, file = 'all', *, cover_image_kwargs:dict = {}, playlist_contents_kwargs:dict = {}):
        '''
        Download all relevant data (playlist metadata, playlist contents, and playlist cover images).

        Parameters
        ----------
        file : {'all', 'imgs', 'mpls', 'artist_genres', 'artists', 'tracks', 'playlists'}
            The file to download. If 'all', all files will be downloaded. Note that due to
            how information is packaged on the spotify API, downloading any of 'artists', 'artist_genres',
            'tracks', or 'mpls' will also download all of the others.
        cover_image_kwargs : dict, optional
            Optional keyword arguments to pass to the `download_playlist_cover_images` method.
        playlist_contents_kwargs : dict, optional
            Optional keyword arguments to pass to the `download_playlist_contents` method.
        '''

        if file == 'all':
            self.download_playlist_metadata()
            self.download_playlist_cover_images(**cover_image_kwargs)
            self.download_playlist_contents(**playlist_contents_kwargs)
            return

        if 'playlists' in file:
            self.download_playlist_metadata()
        elif file == 'imgs':
            self.download_playlist_cover_images(**cover_image_kwargs)
        elif 'img_' in file:
            self.download_playlist_cover_image()
        elif any([fname in file for fname in ('mpl', 'artist', 'tracks')]):
            self.download_playlist_contents(**playlist_contents_kwargs)
    
    def _remove_downloads(self, *, yes_im_sure = False):
        '''
        Delete downloaded files. Will only do it if you really mean it.

        Parameters
        ----------
        yes_im_sure : bool
            Defaults to False. If True, will remove all downloaded data. Otherwise does nothing.
        
        Returns
        -------
        True
        '''

        if yes_im_sure:
            # ok sure
            for file in self.data_files + self.mpl_files + self.img_files:
                if os.path.exists(file):
                    os.remove(file)
        
        return True

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ################################# TABLE PROPERTIES ################################
    @property
    def img_files(self):
        return [os.path.join(os.path.abspath(self.img_dir), x) for x in os.listdir(self.img_dir)]
    
    @property
    def mpl_files(self):
        return [os.path.join(os.path.abspath(self.mpl_dir), x) for x in os.listdir(self.mpl_dir)]
    
    def check_downloaded(self, file:str = 'all'):
        '''
        Check if the requested file has been downloaded, returning the result as a bool.

        Parameters
        ----------
        file : {'all', 'mpls', 'imgs, 'playlists', 'tracks', 'artists', 'artist_genres'}
            The file to check. Will also accept filenames with the .csv extension, or specific mpl/img files (these must have the .csv extension).
            If 'all' (default), will check for all data files, 
            returning True only if all are present. If 'mpl', will check if 
            the mpl directory (`mpl_dir`) is empty. Will also accept a playlist csv
            to check (e.g. 'mpl_2018_01' for 'mpl_2018_01.csv').
            Full filepaths will also be accepted.

        Returns
        -------
        True
            If file is downloaded.

        Raises
        ------
        FileNotFoundError
            If the requested file(s) is/are not found.
        ValueError
            If file is not one of {'all', 'mpls', 'imgs', 'playlists.csv', 'tracks.csv', 'artists.csv', 'artist_genres.csv'}, or a valid mpl_.csv file.
        '''

        filepaths = {
            **{os.path.split(file)[-1].removesuffix('.csv'): file for file in self.data_files},
            **{os.path.split(file)[-1]: file for file in self.data_files}
        }

        if file == 'all':
            out = True
            for file in self.data_files +['mpls', 'imgs']:
                out &= self.check_downloaded(file)
            
            return out
        
        if file == 'mpls':
            if len(os.listdir(self.mpl_dir)) == 0:
                raise FileNotFoundError(f'mpls directory ({self.mpl_dir}) is empty. Try downloading data with the `.download` method.')
            return True
        
        if file == 'imgs':
            if len(os.listdir(self.mpl_dir)) == 0:
                raise FileNotFoundError(f'mpls directory ({self.mpl_dir}) is empty. Try downloading data with the `.download` method.')
            return True
        
        if file in self.img_files + self.mpl_files:
            # dynamically calculated file lists - if it's in there, it exists.
            return True
        
        # in all other cases we must manually check the file
        if file in self.data_files:
            filepath = file
        elif file in filepaths:
            filepath = filepaths[file]
        elif 'mpl_' in file:
            if len(file) != 15: # mpl filenames are 15 chars long
                raise ValueError(f"Cannot check for unrecognised file '{file}'. Must be one of 'all', 'imgs', 'mpls', {set(filepaths)}, a valid mpl_*.csv, cover_*.jpeg, or the full filepath of any of these files.")
            filepath = os.path.join(self.mpl_dir, file)
        elif 'img_' in file:
            if len(file) != 18: # mpl filenames are 15 chars long
                raise ValueError(f"Cannot check for unrecognised file '{file}'. Must be one of 'all', 'imgs', 'mpls', {set(filepaths)}, a valid mpl_*.csv, cover_*.jpeg, or the full filepath of any of these files.")
            filepath = os.path.join(self.img_dir, file)
        elif file not in filepaths:
            raise ValueError(f"Cannot check for unrecognised file '{file}'. Must be one of 'all', 'imgs', 'mpls', {set(filepaths)}, a valid mpl_*.csv, cover_*.jpeg, or the full filepath of any of these files.")
        else:
            filepath = filepaths[file]

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'{file} not found. Try downloading data with the `.download` method.')
        else:
            return True
    
    @property
    def df_tracks(self) -> pd.DataFrame:
        '''
        Convenience property to read the tracks csv to a DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame of all tracks in all monthly playlists.
        '''
        self.check_downloaded(self.tracks_file)

        df = pd.read_csv(self.tracks_file, index_col='id')
        df['release_date'] = pd.to_datetime(df['release_date'], format = r'%Y-%m-%d').dt.date
        return df

    @property
    def df_playlists(self) -> pd.DataFrame:
        '''
        Convenience property to read the playlists metadata from csv to DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame of monthly playlists metadata.
        '''
        self.check_downloaded(self.playlists_file)

        df = pd.read_csv(self.playlists_file, index_col='id')
        df['date'] = pd.to_datetime(df['date'], format = r'%Y-%m-%d').dt.date
        return df

    @property
    def df_artists(self) -> pd.DataFrame:
        '''
        Convenience property to read the artists data from csv to DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing artist information (genres and popularities).
        '''
        self.check_downloaded(self.artists_file)

        return pd.read_csv(self.artists_file, index_col='id')

    @property
    def df_artist_genres(self) -> pd.DataFrame:
        '''
        Convenience property to read the artists data from csv to DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing artist information (genres and popularities).
        '''
        self.check_downloaded(self.artist_genres_file)

        return pd.read_csv(self.artist_genres_file, index_col='id').pivot(columns = 'genre', values = 'genre').map(lambda x: True, na_action='ignore').fillna(False).infer_objects(copy=False)

    def playlist_file(self, identifier:Union[str, datetime.date]) -> str:
        '''
        Filepath for the playlist csv corresponding to the given identifier as a DataFrame. 

        Parameters
        ----------
        identifier : str or datetime.date
            The identifier of the playlist - either a spotify playlist ID, date, or playlist name.
        
        Returns
        -------
        str
            Filepath of the playlist csv.
        '''
        date = self.convert_playlist_identifier(identifier, 'date')

        return os.path.join(self.mpl_dir, f'mpl_{date.year}_{date.month:02d}.csv')

    def df_playlist(self, identifier:Union[str, datetime.date] = 'all') -> pd.DataFrame:
        '''
        Read the playlist csv corresponding to the given identifier as a DataFrame.

        Parameters
        ----------
        identifier : str or datetime.date
            The identifier of the playlist - either a spotify playlist ID, date, or playlist name.
            If 'all' (defualt), will return all playlists in one dataframe, with a multi-index of
            (playlist_id, track position ('id'))
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the track ids in the playlist and the date they were added.
        '''
        if identifier == 'all':
            df = self.sql(f'''
                    SELECT 
                        strptime(CONCAT(filename[-11:-5],'_01'), '%Y_%m_%d') as playlist_id,
                        id,
                        date_added, 
                        track
                    FROM read_csv('{self.mpl_dir}/*.csv', filename = true)'''
            )
            df['playlist_id'] = df['playlist_id'].dt.date.map(self.playlist_dates_to_ids)
            return df.set_index(['playlist_id','id'])


        filepath = self.playlist_file(identifier)
        self.check_downloaded(filepath)
        
        return pd.read_csv( filepath, index_col='id')
    
    def playlist_cover_image(self, identifier:Union[str, datetime.date] = None) -> np.ndarray:
        '''
        Read the cover image for the playlist corresponding to the given identifier.

        Parameters
        ----------
        identifier : str or datetime.date
            The identifier of the playlist - either a spotify playlist ID, date, or playlist name.
        
        Returns
        -------
        numpy.ndarray
            Numpy array containing the image data.
        '''

        date = self.convert_playlist_identifier(identifier, 'date')

        return plt.imread(
            os.path.join(self.img_dir, f'cover_{date.year}_{date.month:02d}.jpeg')
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #################################### CONVERSION ####################################
    @property
    def playlist_ids_to_dates(self) -> dict:
        '''
        Dictionary mapping all monthly playlist IDs to their dates.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist IDs to their dates.
        '''
        return self.df_playlists['date'].to_dict()

    @property
    def playlist_ids_to_names(self) -> dict:
        '''
        Dictionary mapping all monthly playlist IDs to their names.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist IDs to their names.
        '''
        return self.df_playlists['name'].to_dict()
    
    @property
    def playlist_names_to_ids(self) -> dict:
        '''
        Dictionary mapping all monthly playlist names to their IDs.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist names to their IDs.
        '''
        return self.df_playlists.reset_index().set_index('name')['id'].to_dict()
    
    @property
    def playlist_dates_to_ids(self) -> dict:
        '''
        Dictionary mapping all monthly playlist dates to their IDs.

        Returns
        -------
        dict
            A dictionary mapping all monthly playlist dates to their IDs.
        '''
        return self.df_playlists.reset_index().set_index('date')['id'].to_dict()
    
    def _identify_identifier(self, identifier:Union[str, datetime.date]):
        '''
        Identify whether a given identifier is a spotfy id, date, or playlist name.
        
        Parameters
        ----------
        identifier : str or datetime.date
            The identifier whose type will be identified. `datetime.datetime` values will be treated as
            `datetime.date`s, and converted when returned.

        Returns
        -------
        identifier_type : {'date', 'id', 'name'}
            The type of the identifier, returned as a string.
        identifier : str or datetime.date
            The given identifier, converted to the expected type (datetime.date for 'date', and str otherwise).
        
        Raises
        ------
        TypeError
            When identifier is not a str or datetime.date/datetime.datetime.
        '''
        
        if isinstance(identifier, datetime.date):
            return 'date', identifier
        elif isinstance(identifier, datetime.datetime):
            return 'date', identifier.date()
        elif isinstance(identifier, str):
            if identifier[:2] == '20' and identifier.count('-') == 2 and len(identifier) in (8, 9, 10):
                date_split = identifier.split('-')
                return 'date', datetime.date(int(date_split[0]), int(date_split[1]), int(date_split[2]))
            elif identifier[:2] == '20' and identifier.count('-') == 1 and len(identifier) in (6, 7):
                date_split = identifier.split('-')
                return 'date', datetime.date(int(date_split[0]), int(date_split[1]), 1)
            elif identifier in self.playlist_ids:
                return 'id', identifier
            else:
                return 'name', identifier
        else:
            raise TypeError(f'identifier must be either str or datetime.date. Got {type(identifier)}')

    def convert_playlist_identifier(self, identifier:Union[str, datetime.date], return_type:str = 'id') -> str:
        '''
        Parse a given playlist identifier (either playlist date or playlist name) and return
        the corresponding playlist id.

        Parameters
        ----------
        identifier : str or datetime.date
            The identifier to parse. Should be a valid date (either datetime.date or a str in the form YYYY-MM-01),
            or a valid playlist name from `.playlist_names`
        return_type : {'id', 'date', 'name'}, optional
            The type of identifier to return. Defaults to 'id', returning the playlist id.

        Returns
        -------
        id : str
            The identified playlist id
        
        Raises
        ------
        ValueError
            If the identifier cannot be found in the list of known playlist names, and is not in the expected 
            YYYY-MM-01 format.
        '''
        
        identifier_type, identifier = self._identify_identifier(identifier)
        if identifier_type == return_type:
            return identifier
        else:
            if identifier_type == 'id':
                sp_id = identifier
            elif identifier_type == 'date':
                sp_id = self.playlist_dates_to_ids[identifier]
            else:  # identifier_type == 'name'
                sp_id = self.playlist_names_to_ids[identifier]
            
        if return_type == 'id':
            return sp_id
        elif return_type == 'date':
            return self.playlist_ids_to_dates[sp_id]
        else:  # identifier_type == 'name'
            return self.playlist_ids_to_names[sp_id]
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ################################ IDENTIFIER LISTS #################################
    ####### PLAYLISTS #######
    @property
    def playlist_ids(self) -> list:
        '''
        Get a list of all monthly playlist IDs.

        Returns
        -------
        list
            A list of all monthly playlist Spotify IDs.
        '''
        return self.df_playlists.index.tolist()

    @property
    def playlist_names(self) -> np.ndarray:
        '''
        Names of all playlists present in the playlists.csv file.

        Returns
        -------
        np.ndarray
            An array of the names the monthly playlists.
        '''
        return self.df_playlists['name'].unique()
    
    @property
    def playlist_dates(self) -> np.ndarray:
        '''
        Dates of all playlists present in the downloaded playlists.csv file

        Returns
        -------
        np.ndarray
            An array of the dates (months) the monthly playlist data covers
        '''
        return self.df_playlists['date'].unique()

    ####### TRACKS #######
    @property
    def track_ids(self) -> np.ndarray:
        '''
        Array of all artist ids present in all monthly playlists

        Returns
        -------
        np.ndarray
            An array of all artist ids present in all monthly playlists.
        '''
        return self.df_tracks.index.to_numpy()

    @property
    def track_names(self) -> np.ndarray:
        '''
        Array of all artist names present in all monthly playlists

        Returns
        -------
        np.ndarray
            An array of all artist names present in all monthly playlists.
        '''
        return self.df_tracks['name'].values

    ####### ARTISTS #######
    @property
    def artist_ids(self) -> np.ndarray:
        '''
        Array of all artist ids present in all monthly playlists

        Returns
        -------
        np.ndarray
            An array of all artist ids present in all monthly playlists.
        '''
        return self.df_artists.index.to_numpy()

    @property
    def artist_names(self) -> np.ndarray:
        '''
        Array of all artist names present in all monthly playlists

        Returns
        -------
        np.ndarray
            An array of all artist names present in all monthly playlists.
        '''
        return self.df_artists['name'].values
    
    ####### GENRES #######
    @property
    def genres(self) -> np.ndarray:
        '''
        Array of genres present in all monthly playlists

        Returns
        -------
        numpy.ndarray
            Array of all genres present in all monthly playlists.
        '''
        return self.df_artist_genres.columns.to_numpy()

    @property
    def supergenres(self) -> np.ndarray:
        '''
        Array of defined supergenres present in all monthly playlists

        Returns
        -------
        numpy.ndarray
            Array of the pre-defined supergenres.
        '''
        return np.array(list(self.supergenre_lists.keys()))
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #################################### ANALYSIS ####################################
    def sql(self, query: str, *, alias: str = "", params: object = None) -> pd.DataFrame:
        '''
        Query the downloaded data using a sql query, and return in a pandas DataFrame.
        Queries are made via a DuckDB memory connection. Quack quack.

        Parameters
        ----------
        query : str
            The query, passed to duckdb.connection(':memory:').sql().
        alias : str
            No idea really. DuckDB docs are terrible on this. But it's a parameter so I'm givimg you
            access to it!
        params : object
            Again, no idea. See DuckDB docs maybe? You might have more luck than me.
        
        Returns
        -------
        pandas.DataFrame
        '''
        # ensure we have all required data downloaded before querying
        for file in self.data_files + self.img_files + self.mpl_files:
            if file in query:
                self.check_downloaded(file)

        with duckdb.connect(':memory:') as con:
            return con.sql(query=query, alias=alias, params=params).df()

    def artist_timeseries(self) -> pd.DataFrame:
        '''
        A dataframe with index of playlist dates, and columns of artists, with
        values showing how many tracks by/featuring that artist are present in the given months playlist.

        Returns
        -------
        pd.DataFrame
        '''

        sql_text = f'''

        WITH tracks AS (
        SELECT id, artist_1, artist_2, artist_3, artist_4, artist_5, artist_6, artist_7, artist_8, artist_9, artist_10
        FROM read_csv('{self.tracks_file}')
        ),

        playlists AS (
        SELECT 
            track, 
            strptime(CONCAT(filename[-11:-5],'_01'), '%Y_%m_%d') as playlist_date
        FROM read_csv('{self.mpl_dir}/*.csv', filename = true)
        ),

        playlist_artists as(
        SELECT 
            pls.playlist_date, 
            tr.* EXCLUDE id
        FROM playlists pls
        LEFT JOIN tracks tr
        ON tr.id = pls.track
        ),

        artists as(
        SELECT id, name
        FROM read_csv('{self.artists_file}')
        ),

        playlist_artist_stack as (
        SELECT
            playlist_date,
            artist,
        FROM playlist_artists
        UNPIVOT (
            artist FOR artist_col IN (
                artist_1,
                artist_2,
                artist_3,
                artist_4,
                artist_5,
                artist_6,
                artist_7,
                artist_8,
                artist_9,
                artist_10
            )

        )
        ),

        playlist_artist_count as (
        SELECT 
            p.playlist_date, 
            a.name
        FROM playlist_artist_stack p
        JOIN artists a
        ON a.id = p.artist
        )

        PIVOT playlist_artist_count
        ON name
        USING count(*)
        GROUP BY playlist_date
        '''
        
        df = self.sql(sql_text)

        df['playlist_date'] = df['playlist_date'].dt.date
        return df.set_index('playlist_date').sort_index()

    def genre_timeseries(self, *, supergenre = False) -> pd.DataFrame:
        '''
        A dataframe with index of playlist dates, and columns of genres/supergenres, with
        values showing how many tracks by/featuring artists with that genre are in that month's playlist.
        Uses a duckdb backend.

        Parameters
        ----------
        supergenre : bool
            Whether to aggregate genres into supergenres. Defaults to False.

        Returns
        -------
        pd.DataFrame
        '''

        sql_text = f'''{f'SET VARIABLE genre_map = MAP {self.supergenre_map};' if supergenre else ''}
        
        WITH tracks AS (
            SELECT id, artist_1, artist_2, artist_3, artist_4, artist_5, artist_6, artist_7, artist_8, artist_9, artist_10, 
            FROM read_csv('{self.tracks_file}')
        ),

        artists AS (
            SELECT id, 
                {"getvariable('genre_map')[genre] as " if supergenre else ''}genre, 
                true as present
            FROM read_csv('{self.artist_genres_file}')
        ),
        '''

        for i in range(10):
            sql_text += f'''artist_genres_{i+1} AS (
                        SELECT 
                            t.id as track_id, 
                            a.genre,
                            CAST(a.present as int) as present
                        FROM tracks t
                        INNER JOIN artists a
                            ON t.artist_{i+1} = a.id 
                        ),
            
            '''
                
        sql_text += f'''artist_genres AS (
                            SELECT * FROM artist_genres_1
                            UNION SELECT * FROM artist_genres_2
                            UNION SELECT * FROM artist_genres_3
                            UNION SELECT * FROM artist_genres_4
                            UNION SELECT * FROM artist_genres_5
                            UNION SELECT * FROM artist_genres_6
                            UNION SELECT * FROM artist_genres_7
                            UNION SELECT * FROM artist_genres_8
                            UNION SELECT * FROM artist_genres_9
                            UNION SELECT * FROM artist_genres_10
                        ),

                        playlists_genres as(
                        SELECT 
                            strptime(pls.playlist_date, '%Y_%m_%d') as playlist_date, 
                            ag.genre, 
                            ag.present
                        FROM (
                            SELECT 
                                track, 
                                CONCAT(filename[-11:-5],'_01') as playlist_date 
                            FROM read_csv('{self.mpl_dir}/*.csv', filename = true)
                        ) pls
                        LEFT JOIN artist_genres ag
                        ON pls.track = ag.track_id
                        )
                        
                    PIVOT playlists_genres
                        ON genre
                        USING sum(present)
                        GROUP BY playlist_date'''
        
        # quack
        df = self.sql(sql_text).convert_dtypes()
        df['playlist_date'] = df['playlist_date'].dt.date

        return df.set_index('playlist_date').sort_index()
    
    def tracks_with_artist_genre(self, genre, *, search_within_strings = False, playlist = None, supergenre = False):
        '''
        Get all tracks in the monthly playlists that have a given artist genre. The artist genre can
        be a genre (default) or, if `supergenre = True`, a supergenre as specified in `supergenre_lists`.

        Parameters
        ----------
        genre : str
            The genre to search for. Should be one of the available genres (see `genres`) or 
            a supergenre (`supergenres`).
        search_within_strings : bool
            If True, will return tracks whose genre contains the given genre text (LIKE '%genre%'),
            rather than only exact matches. Useful for identifying subgenres.
        playlist : str or datetime.date, optional
            A playlist identifier specifying the playlist to search through. 
            If given, will only return matches that are in the corresponding playlist.
        supergenre : bool, optional
            Whether or not the given genre should be interpreted as a supergenre. Defaults to False.
        '''
        if playlist is not None:
            pl_date = self.convert_playlist_identifier(playlist, 'date')
            pl_fname = f'mpl_{pl_date.year}_{pl_date.month:02d}.csv'
        else:
            pl_fname = '*.csv'

        sql_text = f'''{f'SET VARIABLE genre_map = MAP {self.supergenre_map};' if supergenre else ''}
        
        WITH tracks AS (
            SELECT *
            FROM read_csv('{self.tracks_file}')
        ),

        artists AS (
            SELECT id, 
                {"getvariable('genre_map')[genre] as " if supergenre else ''}genre
            FROM read_csv('{self.artist_genres_file}')
            WHERE genre {f"LIKE '%{genre}%'" if search_within_strings else f" = '{genre}'"} 
        ),
        '''

        for i in range(10):
            sql_text += f'''artist_genres_{i+1} AS (
                        SELECT 
                            t.id as track_id, 
                            t.* EXCLUDE id,
                            a.genre
                        FROM tracks t
                        INNER JOIN artists a
                            ON t.artist_{i+1} = a.id 
                        ),
            
            '''
                
        sql_text += f'''artist_genres AS (
                            SELECT * FROM artist_genres_1
                            UNION SELECT * FROM artist_genres_2
                            UNION SELECT * FROM artist_genres_3
                            UNION SELECT * FROM artist_genres_4
                            UNION SELECT * FROM artist_genres_5
                            UNION SELECT * FROM artist_genres_6
                            UNION SELECT * FROM artist_genres_7
                            UNION SELECT * FROM artist_genres_8
                            UNION SELECT * FROM artist_genres_9
                            UNION SELECT * FROM artist_genres_10
                        )

                        
                        SELECT 
                            strptime(pls.playlist_date, '%Y_%m_%d') as playlist_id, 
                            ag.*
                        FROM (
                            SELECT 
                                track, 
                                CONCAT(filename[65:71],'_01') as playlist_date 
                            FROM read_csv('{self.mpl_dir}/{pl_fname}', filename = true)
                        ) pls
                        INNER JOIN artist_genres ag
                        ON pls.track = ag.track_id
                        '''
        
        # quack
        df = self.sql(sql_text).convert_dtypes()
        df['playlist_id'] = df['playlist_id'].dt.date.map(self.playlist_dates_to_ids)

        return df.set_index(['playlist_id', 'track_id']).sort_index()
        
    def n_tracks_with_artist_genre(self, genre, *, distinct = False, search_within_strings = False, playlist = None, supergenre = False):
        '''
        The number of tracks in the monthly playlists that have a given artist genre. The artist genre can
        be a genre (default) or, if `supergenre = True`, a supergenre as specified in `supergenre_lists`.

        Parameters
        ----------
        genre : str
            The genre to search for. Should be one of the available genres (see `genres`) or 
            a supergenre (`supergenres`).
        distinct : bool
            If True, will not count repeats (e.g. if a track is in multiple playlists).
        search_within_strings : bool
            If True, will return tracks whose genre contains the given genre text (using LIKE '%genre%'),
            rather than only exact matches. Useful for identifying subgenres.
        playlist : str or datetime.date, optional
            A playlist identifier specifying the playlist to search through. 
            If given, will only count matches that are in the corresponding playlist.
        supergenre : bool, optional
            Whether or not the given genre should be interpreted as a supergenre. Defaults to False.
        '''
        df = self.tracks_with_artist_genre(
            genre, 
            search_within_strings=search_within_strings, 
            playlist = playlist,
            supergenre = supergenre
        )
        if distinct:
            return df.groupby('track_id').ngroups
        else:
            return df.groupby(['playlist_id', 'track_id']).ngroups
    
    def genre_track_counts(self, distinct = False, playlist = None, supergenre = False) -> pd.Series:
        '''
        Get the number of tracks of each genre.

        
        Parameters
        ----------
        distinct : bool
            If True, will not count repeats (e.g. if a track is in multiple playlists).
        playlist : str or datetime.date, optional
            A playlist identifier specifying the playlist to search through. 
            If given, will only count matches that are in the corresponding playlist.
        supergenre : bool, optional
            Whether or not to use supergenres. Defaults to False.

        Returns
        -------
        pd.Series
            Series of track counts indexed by genre names
        '''
        # TODO: less silly implementation that doesn't take upwards of 40 seconds to run
        if supergenre:
            return pd.Series({genre: self.n_tracks_with_artist_genre(genre, distinct=distinct, playlist=playlist, supergenre=True) for genre in self.supergenres}, name = 'count').sort_values(ascending=False)
        
        return pd.Series({genre: self.n_tracks_with_artist_genre(genre, distinct=distinct, playlist=playlist) for genre in self.genres}, name = 'count').sort_values(ascending=False)

    @property
    def genres_not_in_supergenres(self) -> set:
        '''
        Array of genres that do not fall into any of the defined supergenre categories.
        Any genre in here will require classification to be included in analysis functions
        using the supergenres.

        Returns
        -------
        set
            Set of genres. May (hopefully) be empty.
        '''
        genres = []
        for g_list in self.supergenre_lists.values():
            genres += g_list

        return set(self.genres) - set(genres)

    def artist_counts(self, playlist = None) -> pd.Series:
        '''
        Series of artist name: number of times that artist appears in `df_tracks`.

        Returns
        -------
        pd.Series
        '''
        
        if playlist is not None:
            pl_date = self.convert_playlist_identifier(playlist, 'date')
            pl_fname = f'mpl_{pl_date.year}_{pl_date.month:02d}.csv'
        else:
            pl_fname = '*.csv'


        sql_text = f'''

            WITH tracks AS (
            SELECT id, artist_1, artist_2, artist_3, artist_4, artist_5, artist_6, artist_7, artist_8, artist_9, artist_10
            FROM read_csv('{self.tracks_file}')
            ),

            playlists AS (
            SELECT 
                track, 
                strptime(CONCAT(filename[-11:-5],'_01'), '%Y_%m_%d') as playlist_id
            FROM read_csv('{self.mpl_dir}/{pl_fname}', filename = true)
            ),

            playlist_artists as(
            SELECT 
                pls.playlist_date, 
                tr.* EXCLUDE id
            FROM playlists pls
            LEFT JOIN tracks tr
            ON tr.id = pls.track
            ),

            artists as(
            SELECT id, name
            FROM read_csv('{self.artists_file}')
            ),

            playlist_artist_stack as (
                SELECT
                    playlist_date,
                    artist,
                FROM playlist_artists
                UNPIVOT (
                    artist FOR artist_col IN (
                        artist_1,
                        artist_2,
                        artist_3,
                        artist_4,
                        artist_5,
                        artist_6,
                        artist_7,
                        artist_8,
                        artist_9,
                        artist_10
                    )

                )
            )

        
            SELECT 
                p.artist as id,
                a.name,
                p.track_count
            FROM (
                SELECT artist, COUNT(artist) as track_count
                FROM playlist_artist_stack
                GROUP BY artist
            ) p
            JOIN artists a
            ON a.id = p.artist
            '''

        return self.sql(sql_text).convert_dtypes().set_index('id').sort_values('track_count', ascending=False)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ################################### PLOTTING  ###################################
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
        df_pl = self.df_playlists
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
                try:
                    im = self.playlist_cover_image(date)
                except FileNotFoundError:
                    im = None
                if im is not None:
                    ax.imshow(self._centre_square_crop(im))

                ax.axis('off')
        
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

        return fig, axes
    
    @staticmethod
    def _plot_ranking(df_rank, title = '', 
                      legend_title = '', legend_style = 'full',
                      highlight = None, highlight_marker = 'o', highlight_colour = 'r',
                      marker = 'x', cmap = 'tab20', marker_colour = 'cmap',
                      figsize = (20,10), xaxis_pad = pd.Timedelta('30D'),
                      label_all_yticks = False):
        '''
        Plot the given ranking timeseries.

        Parameters
        ----------
        df_rank : pandas.DataFrame
            The processed ranking data to plot. Should have an index of dates and columns of the
            item being ranked (e.g. artists, genres, supergenres)
        legend_title : str
            Title for the legend. Ignored if legend_style is None.
        legend_style : {'full', 'partial', None}.
            The style of the legend. 
                'full': all columns will be labelled. 
                'partial': only the highlighted column will be labelled.
                `None`: no legend. 
        highlight : str, optional
            The column to highlight.
        highlight_marker : str, optional
            The marker to use for the highlighted column.
        highlight_colour : str, optional
            The colour to use for the highlighted column.
        marker : str, optional
            The marker to use for the non-highlighted columns.
        cmap : str
            The colormap to use for the non-highlighted values. Must be a recognised matplotlib colourmap.
        marker_colour : str, optional
            The colour to use for the non-highlighted columns. If 'cmap' (default), 
            will use the colour from the colormap.
        figsize : tuple
            The size of the figure.
        xaxis_pad : pandas.Timedelta
            The amount of padding to add to either side of the xaxis, in units of time.
        label_all_yticks : bool
            If True, labels all yticks. Can get messy with large numbers of columns.

        Returns
        -------
        fig, ax
        '''
        
        if highlight is None:
            highlight = ''

        # Plot =) 
        fig, ax = plt.subplots(figsize = figsize)
        cmap = plt.get_cmap('tab20')

        for i, col in enumerate(df_rank):
            if col.lower() == highlight.lower():
                ax.plot(
                    df_rank[col], 
                    color = highlight_colour, 
                    label = col,
                    linestyle = 'None', 
                    marker = highlight_marker
                )
            else:
                ax.plot(
                    df_rank[col], 
                    color = cmap(i) if marker_colour == 'cmap' else marker_colour, 
                    label = col if legend_style.lower() == 'full' else None, 
                    alpha = 1 if highlight == '' else 0.7,
                    linestyle = 'None', 
                    marker = marker
                )
        if label_all_yticks:
            ax.set_yticks(range(1,int(np.max(df_rank) + 1)))

        ax.set_xlim(df_rank.index.min() - xaxis_pad, df_rank.index.max() + xaxis_pad)
        ax.invert_yaxis()
        if legend_style.lower() in ('full', 'partial'):
            ax.legend(bbox_to_anchor = (1, 0.95), title = legend_title)
        ax.set_xlabel('Month')
        ax.set_ylabel('Rank')
        fig.set_constrained_layout(True)

        # set axis box off
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.xaxis.set_minor_locator(m_dates.MonthLocator())
        ax.set_title(title, fontsize = 15)

        return fig, ax

    def plot_genres_rank(self, *,
                         supergenre = True,
                         highlight = None, 
                         figsize = (20,5),
                         xaxis_pad = pd.Timedelta('30D')):
        '''
        Plot the ranking of each genre against time. Ranking of a genre is calculated using the number of
        songs by/featuring an artist that plays that genre in each month's playlist.

        Parameters
        ----------
        supergenre : bool
            Whether to aggregate the genres using the supergenre definitions. Setting to False is not recommended,
            as the large number of base genres makes the plot hard to interpret.
        highlight : str, optional
            The name of the genre/supergenre to highlight in red.
        figsize : tuple
            The size of the figure, in inches.
        xaxis_pad : pandas.Timedelta
            The amount of padding to add to either side of the xaxis, in units of time.
        '''

        df_genres = self.genre_timeseries(supergenre=supergenre)

        # Calculate ranks in each month
        df_rank = df_genres.replace(0, pd.NA).rank(
            ascending=False, 
            method = 'first', 
            na_option='keep',
            axis = 1
        ).sort_values(
            axis = 1, 
            by = df_genres.index.values[0],
            ascending=True
        )

        return self._plot_ranking(
            df_rank,
            title = f'{"super" if supergenre else ''}genre rankings'.capitalize(),
            legend_title = f'{"Super" if supergenre else ''}genres'.capitalize(),
            legend_style = 'full' if supergenre else 'partial',
            marker_colour = 'cmap' if supergenre else 'tab:cyan',
            highlight=highlight,
            figsize=figsize,
            xaxis_pad=xaxis_pad,
            label_all_yticks=supergenre
        )

    def plot_artists_rank(self, *,
                         highlight = None, 
                         figsize = (25,5),
                         xaxis_pad = pd.Timedelta('30D')):
        '''
        Plot the ranking of each artist against time. Ranking of an artist is calculated using the number of
        songs by/featuring that artist each month's playlist.

        Parameters
        ----------
        highlight : str, optional
            The name of the genre/supergenre to highlight in red.
        figsize : tuple
            The size of the figure, in inches.
        xaxis_pad : pandas.Timedelta
            The amount of padding to add to either side of the xaxis, in units of time.
        '''

        df_artists = self.artist_timeseries()

        # Calculate ranks in each month
        df_rank = df_artists.replace(0, pd.NA).rank(
            ascending=False, 
            method = 'first', 
            na_option='keep',
            axis = 1
        ).sort_values(
            axis = 1, 
            by = df_artists.index.values[0],
            ascending=True
        )

        return self._plot_ranking(
            df_rank,
            title = 'Artist rankings',
            legend_title = 'Artists',
            legend_style = 'partial',
            marker_colour = 'tab:cyan',
            marker = 'o',
            highlight=highlight,
            figsize=figsize,
            xaxis_pad=xaxis_pad,
            label_all_yticks=False
        )

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
        
        df_mpls = MonthlyPlaylistHandler().df_playlists
        if sp_id:
            data = df_mpls.loc[sp_id]
        elif date:
            data = df_mpls.loc[df_mpls['date'] == pd.to_datetime(date).date()].iloc[0]
        elif name:
            data = df_mpls.loc[df_mpls['name'] == name].iloc[0]

        for item in data.index:
            setattr(self, item, data[item])
        self.sp_id = data.name

    @classmethod
    def _identify_identifier(cls, identifier):
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
