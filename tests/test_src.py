import os
from spmpls.playlists import MonthlyPlaylistHandler, LoggingSpotifyClient, DATA_DIR
from spotipy import Spotify, SpotifyOAuth
import pytest

def ensure_data_present():
    '''If data is not present, download it. Otherwise, work with what we have.'''
    
    mpl = MonthlyPlaylistHandler()

    for file in ['playlists', 'imgs', 'mpls', 'artists', 'artist_genres', 'tracks']:
        try:
            mpl.check_downloaded(file)
        except FileNotFoundError:
            mpl.download(file)
    
def test_download():
    pass

def test_file_read():
    pass

def test_timeseries():
    pass

@pytest.fixture(autouse=True)
def clean():
    '''Clean up after all tests'''
    # get contents of data directory
    files_before_test = set(os.listdir(DATA_DIR))

    # run test
    yield

    # get new contents of data dir
    files_after_test = set(os.listdir(DATA_DIR))

    # raise an error if we've deleted any files so we can cry I guess?
    assert not len(files_after_test) < len(files_before_test)
    
    new_files = files_after_test - files_before_test
    if len(new_files) > 0:
        for filename in new_files:
            os.remove(
                os.path.join(DATA_DIR, filename)
            )
    
    # assert we have successfully tidied up
    assert files_before_test == set(os.listdir(DATA_DIR))


