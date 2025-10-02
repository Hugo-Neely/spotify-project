import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import datetime
from typing import Union, Tuple
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt

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
    
    data_dir = os.path.abspath('data')
    
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
            spotify_client = self._new_sp_client()
        self.sp_client = spotify_client

    def _new_sp_client(self, client_id:str = None, client_secret:str = None, redirect_uri:str = None,
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
        
        return spotipy.Spotify(
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
            playlists = self.sp_client.current_user_playlists(offset = offset)
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
        df['cover_image_url'] = df['images'].apply(lambda x: x[0]['url'] if not pd.isna(x) else x)

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

    def get_playlist_cover_image(self, playlist_id:str = None, playlist_date:Union[str, datetime.date] = None,
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


        r = requests.get(pl_im_url.values[0])
        im = plt.imread(BytesIO(r.content), format='jpeg')
        plt.imsave(save_path, im)
        return im
    
    def plot_playlist_covers(self, scale:float = 1.0) -> Tuple[plt.Figure, np.ndarray]:
        '''
        Plot a collage of all monthly playlist cover images.

        Parameters
        ----------
        scale : float
            A scaling factor to apply to the size of each image in the plot. Default is 1.0.
            Applied to the figsize parameter of plt.subplots().
        
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

        fig, axes = plt.subplots(n_months, n_years, figsize=(n_years*scale, n_months*scale))

        for m, months_ax in enumerate(axes):
            for y, ax in enumerate(months_ax):
                date = f'{min_year+y}-{m+1:02d}-01'
                im = self.get_playlist_cover_image(playlist_date=date, errors='ignore')
                if im is not None:
                    ax.imshow(self._make_im_square(im))

                ax.axis('off')
        
        fig.subplots_adjust(wspace=0, hspace=0)

        return fig, axes
    
    @staticmethod
    def _make_im_square(im):
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





        
