import pickle
import matplotlib.pyplot as plt
import os
import subprocess
import glob

def generate_video(solution):
    for i in range(len(solution)):
        heatmap = solution[i].untensorized()
        plt.imshow(heatmap)
        plt.savefig("video" + "/file%02d.png" % i)

    os.chdir("video")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

solution = pickle.load(open("data.pk", "rb"))
generate_video(solution)
