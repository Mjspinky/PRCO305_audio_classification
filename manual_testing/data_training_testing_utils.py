from utils.data_processing_utils import create_data_file, create_data
from datetime import datetime


def test_csv_data_creation():
    data_goes_here = '../processed_data/recorded_data.csv'
    create_data_file(data_goes_here)

    song_name = '../utils/current_recording.au'
    create_data(song_name, song_name, 'test', data_goes_here)


def get_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

# test_csv_data_creation()
