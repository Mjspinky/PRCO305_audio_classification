from pydub import AudioSegment
import os
import ffmpeg
import re


# TODO:Properly go through each salsa track and determine if its salsa, bachata, etc
# TODO:Find more music on fma adding in Cha Cha, Bachata and Kizomba to the genre lists

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
        dst = f"{genre}." + str(count + 1) + ".mp3"
        src = f'./genres/{genre}/' + filename
        dst = f'./genres/{genre}/' + dst

        os.rename(src, dst)


def check_for_new_songs():
    flag = 0
    for folder in os.listdir(f'./genres/'):
        folder_name_length = len(folder)
        pattern = f'[{folder}]{folder_name_length}[0-9]+.au'
        for filename in os.listdir(f'./genres/{folder}/'):
            if not re.match(pattern, filename):  # TODO:Check this regex [\{genre\}]{len(folder)}[0-9]+.au
                file_rename(folder)
                audio_clipping(folder)
                mp3_to_au_convert(folder)
                flag = 1
        if flag == 1:
            delete_unnecessary_files(folder)


check_for_new_songs()
