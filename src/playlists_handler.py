import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import datetime

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
    
    data_dir = 'data'
    
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
                       backoff_factor:float = 0.5, **kwargs):
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
        
    def get_monthly_playlists(self, to_csv:bool = True) -> pd.DataFrame:
        '''
        Get a dataframe containing monthly playlists metadata.

        Parameters
        ----------
        to_csv : bool
            If True, will save the resulting pd.DataFrame as a CSV in the data directory.
            Filenames are in the format 'playlists_YYYY-MM-DD.csv'. Note that this means any
            other files created on this day will be overwritten.
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

        df['n_tracks'] = df['tracks'].apply(lambda x: eval(x)['total'])

        # only keep useful columns
        df = df[['id', 'name', 'description', 'n_tracks', 'href', 'images', 'snapshot_id']].set_index('id')

        # save if requested, and return
        if to_csv:
            df.to_csv(
                os.path.join(self.data_dir, f'playlists_{str(datetime.datetime.now().date())}.csv')
            )
        return df

    @property
    def latest_playlists_file(self):
        # get the most recent date by extracting the date part of each playlist_DATE.csv file.
        # then convert to datetime, and get the most recent (max)
        most_recent_date = pd.to_datetime([file.replace('playlists_','').replace('.csv','') for file in os.listdir(self.data_dir) if 'playlists' in file]).max()

        return f'playlists_{str(most_recent_date.date())}.csv'

    def read_monthly_playlists(self, date = 'latest'):
        '''
        Read a saved DataFrame of monthly playlists.

        Parameters
        ----------
        date : str or datetime.date
            The date the data was collected. Defaults to 'latest', which retrieves the most recent
            file. If the date is specified as a string, it should be given in the form YYYY-MM-DD.
        
        Returns
        -------
        pandas.DataFrame
        '''
        if date == 'latest':
            file = self.latest_playlists_file
        elif isinstance(date, datetime.date):
            file = f'playlists_{str(date.date())}.csv'
        elif isinstance(date, str):
            file = f'playlists_{date}.csv'
        else:
            raise ValueError(f'Unexpected type "{type(date)}" for date input encountered. Try entering date as either a string (YYYY-MM-DD) or datetime.date.')
        
        return pd.read_csv(
            os.path.join(self.data_dir, file), 
            #index_col = 0,
        ).set_index('id')

    
