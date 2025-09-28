import os
from playlists_handler import MonthlyPlaylistHandler


def test_read_mpl():

    # create new handler
    mpl = MonthlyPlaylistHandler()

    # create a new dummy file, from far in the future
    test_file = 'playlists_2200-01-01.csv'
    with open(os.path.join(mpl.data_dir, test_file), 'w') as f:
        f.write(
'''id,name,description,n_tracks,href,images,snapshot_id
aaaaaa,Nov-2199,testtesttest,500,https://example.com,"[{'height': None, 'url': 'https://example.com', 'width': None}]",bbbbbbbb
aaaaaa,Dec-2199,testtesttest,500,https://example.com,"[{'height': None, 'url': 'https://example.com', 'width': None}]",bbbbbbbb''')

    # check the file we created is identified as the most recent file
    assert mpl.latest_playlists_file == test_file

    # read latest monthly playlists file. check the columns are as expected
    df_pl = mpl.read_monthly_playlists()
    assert df_pl.index.name == 'id'
    assert df_pl.columns.to_list() == ['name', 'description', 'n_tracks', 'href', 'images', 'snapshot_id']
