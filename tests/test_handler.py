import os
import json
from spmpls.playlists import MonthlyPlaylistHandler
import pytest
from dotenv import load_dotenv

@pytest.fixture(scope = 'session', autouse=True)
def setup():
    '''
    Tests the MonthlyPlaylistHandler's ability to download the expected files, and to successfully delete them.
    As part of this, ensures the required environment variables are present.
    '''

    load_dotenv()

    assert os.getenv('SPOTIPY_CLIENT_ID'), "SPOTIPY_CLIENT_ID not found in environment variables."
    assert os.getenv('SPOTIPY_CLIENT_SECRET'), "SPOTIPY_CLIENT_SECRET not found in environment variables."
    assert os.getenv('SPOTIPY_REDIRECT_URI'), "SPOTIPY_REDIRECT_URI not found in environment variables."

    # ensure we have spotify login credentials in cache
    if not os.path.isfile('.cache') and os.getenv("SPOTIPY_REFRESH_TOKEN"):
        cache_data = {
            'access_token': "",  # to be refreshed
            'token_type': "Bearer",
            'expires_in': 3600,
            'scope': "playlist-read-private",
            'expires_at':0,  # refresh now please
            'refresh_token': os.getenv("SPOTIPY_REFRESH_TOKEN")
        }
        with open(".cache", "w") as f:
            json.dump(cache_data, f)

    mpl = MonthlyPlaylistHandler()

    mpl._remove_downloads(yes_im_sure=True)
    data_dir_contents = os.listdir(mpl.data_dir)
    
    # assert all data has been removed
    assert 'playlists.csv' not in data_dir_contents, "playlists.csv not deleted correctly"
    assert 'artist_genres.csv' not in data_dir_contents, "artist_genres.csv not deleted correctly"
    assert 'artists.csv' not in data_dir_contents, "artists.csv not deleted correctly"
    assert 'tracks.csv' not in data_dir_contents, "tracks.csv not deleted correctly"
    assert len(os.listdir(mpl.img_dir)) == 0, "imgs directory not deleted correctly"
    assert len(os.listdir(mpl.mpl_dir)) == 0, "mpls directory not deleted correctly"

    # should download all data
    mpl.download()

    assert mpl.check_downloaded('playlists'), "playlists not downloaded properly"
    assert mpl.check_downloaded('playlists.csv'), "playlists.csv not downloaded properly"

    assert mpl.check_downloaded('artist_genres'), "artist_genres not downloaded properly"
    assert mpl.check_downloaded('artist_genres.csv'), "artist_genres.csv not recognised properly"

    assert mpl.check_downloaded('artists'), " not downloaded properly"
    assert mpl.check_downloaded('artists.csv'), " not downloaded properly"

    assert mpl.check_downloaded('tracks'), "tracks not downloaded properly"
    assert mpl.check_downloaded('tracks.csv'), "tracks.csv not recognised properly"

    # don't have a master list of playlist-specific files, so have to settle for checking the directories aren't empty
    assert mpl.check_downloaded('mpls'), "mpls not downloaded properly"
    assert mpl.check_downloaded('imgs'), "imgs not downloaded properly"

    return

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
