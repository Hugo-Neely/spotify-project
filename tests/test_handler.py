import os
from spmpls.playlists import MonthlyPlaylistHandler
import pytest
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def ensure_downlaoded(request):
    '''Ensure data is downloaded'''

    if 'nodownload' in request.keywords:
        return
    
    mpl = MonthlyPlaylistHandler()

    for file in ['playlists', 'imgs', 'mpls', 'artists', 'artist_genres', 'tracks']:
        try:
            mpl.check_downloaded(file)
        except FileNotFoundError:
            mpl.download(file)


@pytest.fixture(autouse=True)
def load_env():
    '''Load spotipy environment variables'''

    load_dotenv()

    assert os.getenv('SPOTIPY_CLIENT_ID'), "SPOTIPY_CLIENT_ID not found in environment variables."
    assert os.getenv('SPOTIPY_CLIENT_SECRET'), "SPOTIPY_CLIENT_SECRET not found in environment variables."
    assert os.getenv('SPOTIPY_REDIRECT_URI'), "SPOTIPY_REDIRECT_URI not found in environment variables."

@pytest.mark.nodownload
def test_download():
    '''
    Tests the MonthlyPlaylistHandler's ability to download the expected files, and to successfully delete them.
    '''
    
    mpl = MonthlyPlaylistHandler()

    mpl._remove_downloads(yes_im_sure=True)
    data_dir_contents = os.listdir(mpl.data_dir)
    
    # assert all data has been removed
    assert 'playlists.csv' not in data_dir_contents
    assert 'artist_genres.csv' not in data_dir_contents
    assert 'artists.csv' not in data_dir_contents
    assert 'tracks.csv' not in data_dir_contents
    assert len(os.listdir(mpl.img_dir)) == 0
    assert len(os.listdir(mpl.mpl_dir)) == 0

    # should download all data
    mpl.download()

    assert mpl.check_downloaded('playlists')
    assert mpl.check_downloaded('playlists.csv')

    assert mpl.check_downloaded('artist_genres')
    assert mpl.check_downloaded('artist_genres.csv')

    assert mpl.check_downloaded('artists')
    assert mpl.check_downloaded('artists.csv')

    assert mpl.check_downloaded('tracks')
    assert mpl.check_downloaded('tracks.csv')

    # don't have a master list of playlist-specific files, so have to settle for checking the directories aren't empty
    assert mpl.check_downloaded('mpls')
    assert mpl.check_downloaded('imgs')

def test_df_artists():
    '''
    Ensure df_artists has critical columns
    '''

    df_artists = MonthlyPlaylistHandler().df_artists
    assert df_artists.index.name == 'id'
    assert 'name' in df_artists

def test_df_playlists():
    '''
    Ensure df_playlists has critical columns
    '''

    df_playlists = MonthlyPlaylistHandler().df_playlists
    assert df_playlists.index.name == 'id'
    assert 'cover_image_url' in df_playlists
    assert 'date' in df_playlists
    assert 'name' in df_playlists

def test_df_tracks():
    '''
    Ensure df_tracks has critical columns
    '''

    df_tracks = MonthlyPlaylistHandler().df_tracks
    assert df_tracks.index.name == 'id'
    assert 'album_id' in df_tracks
    assert 'artist_1' in df_tracks
    assert 'artist_2' in df_tracks
    assert 'artist_3' in df_tracks
    assert 'artist_4' in df_tracks
    assert 'artist_5' in df_tracks
    assert 'artist_6' in df_tracks
    assert 'artist_7' in df_tracks
    assert 'artist_8' in df_tracks
    assert 'artist_9' in df_tracks
    assert 'artist_10' in df_tracks
