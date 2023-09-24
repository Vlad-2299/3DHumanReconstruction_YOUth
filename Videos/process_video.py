import os
import ffmpeg
import math
import shutil
import argparse
#!pip install ffmpeg-python


#parser = argparse.ArgumentParser(description='Process_Video')
#parser.add_argument('--video', dest='video', required=False, type=str, help='video-name', default="Ola")
#
#args = parser.parse_args()

    
    
#def check_input():
#    if len(args.video):
#        if os.path.isfile(args.video):
#            videofile = args.video
#            return videofile
#        else:
#            raise IOError('Error: --video must refer to a video file, not directory.')

def check_input(vid_path):
    if len(vid_path):
        if os.path.isfile(vid_path):
            videofile = vid_path
            return videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

def crop_video(video, video_width, video_height):
    '''
    Splits the original 4 view viedeo into 4 single view videos
    '''
    video_name = video.split('.mp4')[0]
    video_trim_output = ['_0', '_1', '_2', '_3']

    video_trim_x = [0, video_width/2, 0, video_width/2]
    video_trim_y = [0, 0, video_height/2, video_height/2]
    
    croped_videos = []
    
    for idx, trim_section in enumerate(video_trim_output):
        video_name_output = video_name + trim_section + '.mp4'
        if os.path.exists(video_name_output):
            print(f'Cropped video {video_name_output} already exists!')
            croped_videos.append(video_name_output)
        else:
            command = f'ffmpeg -i {video} -filter:v "crop={video_width/2}:{video_height/2}:{video_trim_x[idx]}:{video_trim_y[idx]}" -c:v libx264 -crf 18 -preset veryfast {video_name_output}'
            os.system(command)
            print(f'{video_name_output}: cropped')
            croped_videos.append(video_name_output)
        
    return croped_videos


def trim_video(video_path, video, start_time, end_time):
    '''
    Trim video. Length should be specified in this method
    start_time: hh:mm:ss - start of trimmed video
    end_time: hh:mm:ss - end of trimmed video
    '''
    video_name = video.split('.mp4')[0]
    data_dir = os.path.dirname(os.path.dirname(video_path))
    folder_name = video_name + '_' + start_time.replace(":", "." ) + '_' + end_time.replace(":", "." )
    trimmed_folder = os.path.join(data_dir, folder_name)

    if not os.path.exists(trimmed_folder):
        os.mkdir(trimmed_folder)
    else:
        shutil.rmtree(trimmed_folder)
        os.mkdir(trimmed_folder)

    out_path = os.path.join(trimmed_folder, video)
    #command = f'ffmpeg -ss 00:03:45.0 -i {video} -c copy -t 00:00:10.0 {video_name_t}'
    #command = f'ffmpeg -ss {init_time} -t {duration_time} -i {video} -async 1 {video_name_t}'
    #command = f'ffmpeg -i {video} -ss 3 -vcodec copy -acodec copy {video_name_t}'
    #command = f'ffmpeg -ss 00:01:00 -to 00:02:00 -i {video} -c copy {video_name_t}'
    #command = f'ffmpeg -ss {start_time} -to {end_time} -i {video} -async 1 {video_name_t}'
    command = f'ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c:v copy -an {out_path}'
    os.system(command)
    if os.path.exists(out_path):
        out = os.path.normpath(out_path)
        print(f'Trimmed {video} was saved to {out}')
        return out
    else:
        IOError('Error: --Video was not cropped successefuly')


def get_frames(video):
    '''
    Splits the mp4 video into frames (30 frames per second)
    '''
    video_name = video.split('.mp4')[0]
    if os.path.exists(video_name):
        shutil.rmtree(video_name)
    
    print(f'Getting frames of {video}')
    os.mkdir(video_name)
    command = f'ffmpeg -i {video} -q:v 1 -vf fps=30/1 {video_name}/%05d.png'
    os.system(command)



def post_folder(folder, f_per_f):
    '''
    Creates one folder for each 100 frames
    '''
    print(f'Post folder: {folder}')
    frame_list = os.listdir(folder)
    n_folders = math.ceil(len(frame_list) / f_per_f)
    subf_list = []
    iter = 0
    last = 0
    N_ = 0
    for n in range(n_folders):
        sub_file = os.path.basename(folder) + '_' + str(n).zfill(2)
        os.mkdir(os.path.join(folder, sub_file))
        subf_list.append(sub_file)

        for frame in frame_list[last:last+f_per_f]:
            iter += 1
            shutil.move(f"{folder}/{frame}", f'{folder}/{sub_file}/{frame}')
        last = iter
        N_ = n
    print(f'{folder} was subdivided into {N_} folders, each with maximum of {f_per_f} frames')



def main_video_prossessor(path, frames_per_folder):
    '''
    Given a video:
        crop 4 views into individual videos
        for each video get 30 frames per second
        for each 1000 frames, save in individual directory
    '''
    #video_name = check_input()
    video_name = check_input(path)

    probe = ffmpeg.probe(path)
    video_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
    video_width = video_streams[0]['width'] # 1920
    video_height = video_streams[0]['height'] #1080
    print(video_name)
    print(f'--- Start Processing {video_name} ---')
    #trim_video(video_name)
    croped_vids = crop_video(video_name, video_width, video_height)
    assert len(croped_vids) == 4, 'Variable should contain the path for the 4 videos of different views!'
    
    for vid in croped_vids:
        vid_name = vid.split('.mp4')[0]
        get_frames(vid)
        post_folder(vid_name, frames_per_folder)

#if __name__ == "__main__":
    #main_video_pross()
    