from pydub import AudioSegment
import os
import ffmpeg
import re


def audio_clipping(filename, genre):
    song = AudioSegment.from_mp3(f'./genres/{genre}/{filename}')
    thirty_seconds = 30 * 1000
    song[:thirty_seconds].export(f'./genres/{genre}/{filename}', format='mp3')


def mp3_to_au_convert(filename, genre):
    stream = ffmpeg.input(f'./genres/{genre}/{filename}')
    length = len(filename)
    filename = filename[0:length - 4] + ".au"
    stream = ffmpeg.output(stream, f'./genres/{genre}/{filename}')
    ffmpeg.run(stream)


def delete_unnecessary_files(genre):
    dir_name = f"./genres/{genre}/"
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(".mp3"):
            os.remove(os.path.join(dir_name, item))


def file_rename(genre):
    for count, filename in enumerate(os.listdir(f'./genres/{genre}/')):
        pattern = f'[{genre}]+.[0-9]+.au'
        if not re.match(pattern, filename):
            dst = f"{genre}." + str(count + 1) + ".au"
            src = f'./genres/{genre}/' + filename
            dst = f'./genres/{genre}/' + dst
            os.rename(src, dst)


def check_for_new_songs():
    flag = 0
    for folder in os.listdir(f'./genres/'):
        pattern = f'[{folder}]+.[0-9]+.au'
        for filename in os.listdir(f'./genres/{folder}/'):
            if not re.match(pattern, filename):
                audio_clipping(filename, folder)
                mp3_to_au_convert(filename, folder)
                flag = 1
        if flag == 1:
            delete_unnecessary_files(folder)
            file_rename(folder)


check_for_new_songs()
