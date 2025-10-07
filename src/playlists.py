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
import matplotlib.pyplot as plt
import logging

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
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.WARNING)
logger_ch.set_name('console')
logger_ch.setFormatter(logger_console_formatter)
logger.addHandler(logger_ch)


class MonthlyPlaylistHandler:
    '''
    Handler for monthly playlist data. Contains a spotipy Spotify instance,
    handling authentication.
    '''

    # style of each monthly playlist title in each year
    year_styles = {'2018':'2018',
                   '2019':'2019',
                   '2020':'2020',
                   '2021':'2021',
                   '2022':' 22',
                   '2023':' 23',
                   '2024':'-24',
                   '2025':'2025'}
    
    data_dir = DATA_DIR
    
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

    def read_monthly_playlists(self, date = None) -> pd.DataFrame:
        '''
        Read a saved DataFrame of monthly playlists.

        Parameters
        ----------
        date : str or datetime.date
            The date the data was collected. If None, will attempt to read the undated file,
            which is assumed to be the most recent. If the undated file does not exist,
            will read the most recent dated file.
            
            If the date is specified as a string, it should be given in the form YYYY-MM-DD.
        
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

    def get_tracks(self):
        '''
        Placeholder for getting tracks in each monthly playlist.
        '''
        pass


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
        near_limit = self.api_rate > (self.RATE_LIMIT_PER_30S * limit_proportion)  # 80% of limit

        if near_limit:
            if warn:
                logger.warning(f'API rate limit exceeded ({self.api_rate} > {self.RATE_LIMIT_PER_30S}), sleeping for {sleep_time} seconds.')
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