import sys
import os
from multiprocessing.pool import ThreadPool

import youtube_dl
import ffmpeg


class VidInfo:
    def __init__(self, yt_id, start_time, tag, mode, outdir):
        self.yt_id = yt_id
        self.start_time = float(start_time)
        self.end_time = self.start_time+10
        self.tag = tag

        out_filename = os.path.join(outdir, mode)
        self.out_filename = os.path.join(out_filename, f'{yt_id}_{self.start_time}.mp4')

# zzpQAtOmMhQ,230,playing clarinet,train
def download(vidinfo):

    yt_base_url = 'https://www.youtube.com/watch?v='

    yt_url = yt_base_url+vidinfo.yt_id
    print(vidinfo.yt_id)
    ydl_opts = {
        # 'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'quiet': True,
        # 'ignoreerrors': True,
        'no_warnings': True,
        'proxy': 'http://127.0.0.1:1087',
        # 'output': '%(id)s.%(ext)s'
     }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            download_url = ydl.extract_info(url=yt_url, download=False)['url']
            # print(download_url.keys())
            # print(download_url)
    except:
        return_msg = '{}, ERROR (youtube)!'.format(vidinfo.yt_id)
        return return_msg
    try:
        (
            ffmpeg
                .input(download_url, ss=vidinfo.start_time, to=vidinfo.end_time)
                .output(vidinfo.out_filename, format='mp4', r=25, vcodec='libx264',
                        crf=18, preset='veryfast', pix_fmt='yuv420p', acodec='aac', audio_bitrate=128000,
                        strict='experimental')
                .global_args('-y')
                .global_args('http_proxy', 'http://127.0.0.1:1087')
                .global_args('-loglevel', 'error')
                .run()

        )
    except:
        return_msg = '{}, ERROR (ffmpeg)!'.format(vidinfo.yt_id)
        return return_msg

    return '{}, DONE!'.format(vidinfo.yt_id)


if __name__ == '__main__':

    # split = sys.argv[1]
    csv_file = '/home/aitical/data4t2/vggsound/vggsound.csv'
    out_dir = '/home/aitical/data4t2/vggsound'

    os.makedirs(os.path.join(out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'test'), exist_ok=True)

    with open(csv_file, 'r') as f:
        lines = f.readlines()
        lines = [x.split(',') for x in lines]
        vidinfos = [VidInfo(x[0], x[1], x[2], x[3], out_dir) for x in lines]

    bad_files = open(os.path.join(out_dir, 'bad_files.txt'), 'w')
    results = ThreadPool(5).imap_unordered(download, vidinfos)
    cnt = 0
    for r in results:
        cnt += 1
        print(cnt, '/', len(vidinfos), r)
        if 'ERROR' in r:
            bad_files.write(r + '\n')
    bad_files.close()