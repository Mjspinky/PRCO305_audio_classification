from pydub import AudioSegment
import os
import ffmpeg

flag = 0


# TODO:Properly go through each salsa track and determine if its salsa, bachata, etc
# TODO:Find more music on fma adding in Cha Cha, Bachata and Kizomba to the genre lists

def audio_clipping(filename):
    song = AudioSegment.from_mp3(f'./genres/salsa/{filename}')
    thirty_seconds = 30 * 1000
    song[:thirty_seconds].export(f'./genres/salsa/{filename}', format='mp3')


def mp3_to_au_convert(filename):
    stream = ffmpeg.input(f'./genres/salsa/{filename}')
    length = len(filename)
    filename = filename[0:length - 4] + ".au"
    stream = ffmpeg.output(stream, f'./genres/salsa/{filename}')
    ffmpeg.run(stream)


def delete_unnecessary_files():
    dir_name = "./genres/salsa/"
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(".mp3"):
            os.remove(os.path.join(dir_name, item))


def file_rename():
    for count, filename in enumerate(os.listdir(f'./genres/salsa/')):
        dst = "salsa." + str(count + 1) + ".mp3"
        src = './genres/salsa/' + filename
        dst = './genres/salsa/' + dst

        os.rename(src, dst)


file_rename()
for filename in os.listdir(f'./genres/salsa/'):
    if filename[-1] == "3":
        audio_clipping(filename)
        mp3_to_au_convert(filename)
        flag = 1

if flag == 1:
    delete_unnecessary_files()
