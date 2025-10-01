import os
from playlists_handler import MonthlyPlaylistHandler
from spotipy import Spotify, SpotifyOAuth
import pytest

DATA_DIR = MonthlyPlaylistHandler.data_dir

def create_test_file(mpl_handler):
    '''
    Create a new test file in the data directory, returning its name
    '''
        # create a new dummy file, from far in the future
    test_file = 'playlists_2200-01-01.csv'
    with open(os.path.join(mpl_handler.data_dir, test_file), 'w') as f:
        f.write(
'''id,name,description,n_tracks,href,images,snapshot_id
aaaaaa,Nov-2199,testtesttest,500,https://example.com,"[{'height': None, 'url': 'https://example.com', 'width': None}]",bbbbbbbb
aaaaaa,Dec-2199,testtesttest,500,https://example.com,"[{'height': None, 'url': 'https://example.com', 'width': None}]",bbbbbbbb''')

    return test_file

def test_read_mpl():

    # create new handler with dummy credentials
    mpl = MonthlyPlaylistHandler(
            Spotify(
                auth_manager=SpotifyOAuth(
                    client_id='test',
                    client_secret='test',
                    redirect_uri='https://www.example.org/callback'
                )
            )
        )

    # create a new dummy file, from far in the future
    test_file = create_test_file(mpl)


    # check the file we created is identified as the most recent file
    assert mpl.latest_playlists_file == test_file

    # read latest monthly playlists file. check the columns are as expected
    df_pl = mpl.read_monthly_playlists()
    assert df_pl.index.name == 'id'
    assert df_pl.columns.to_list() == ['name', 'description', 'n_tracks', 'href', 'images', 'snapshot_id']

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


