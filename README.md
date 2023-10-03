# 3D Human Reconstruction on the Multi-view, In-the-wild, YOUth data
Thesis research framework developed by Vladyskav Kalyuzhnyy, Utrecht Univercity

## Dependencies
Windows or Linux, Python3.9

```bash
pip install -r requirements.txt
```


## Deployment
![figure](/assets/model_overview.png)

**General System Control:**<br>

```general.ipynb```

Main notebook that executes all framework stages <br>

**AlphaPose:**<br>
<i>viz_alphapose_output</i>:general.ipynb debugger variable that saves the visualization AlphaPose output in 'ProcessJSON/data/video_n/Camera_n/viz'

```AlphaPose/scripts/demo_inference.py```

AlphaPose variable controller and 2D joint regression, yolox and Torchreid model deployment

```AlphaPose/alphapose/utils/writer.py```
<i>update</i> method stores a dummy detection instance for frames in which no detections were captured<br>

```AlphaPose/alphapose/utils/pPose_nms.py```
<i>write_json</i> method writes all the AlphaPose output to a JSON file<br>

**Detectron2:**<br>
```Detectron2/process_detectron.py```
Python file which contains all the methods required to execute and process the data from Detectron2<br>

```Detectron2/Detectron2.ipynb```
Stand-alone execution of Detectron2, used for debugging


**DMMR:**<br>
```DMMR/cfg_files```
Stores the yaml configuration file which contains all the stored parameters for the stage of camera calibration and 3D human reconstruction<br>

```DMMR/data/YOUth```
<i>YOUth</i> is a replacement data folder. It should be always present in the data folder. <i>general.ipynb</i> will update this folder with the current video's data<br>

```DMMR/main.py```
Execute <i>main.py</i> for camera calibration and 3d human reconstruction<br>


```DMMR/viz_cameras.py```
Execute <i>viz_cameras.py</i> for a sequential mesh visualization. This scrip requiers the 'output' file to be present in the DMMR folder

![figure](/assets/rec_merge.png)


```DMMR/output```
output data generated from the execution of the <i>main.py</i> file. <br>

output\\
	  |--images: empty file\\
	  |--meshes: mesh objects of each individual in each frame\\
	  |--reprojection: 2D and 3D keyppint information per each frame\\
	  |--results: plk files with fitting information of each frame





## Acknowledgments
[FFmpeg] (https://github.com/FFmpeg/FFmpeg)<br>
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)<br>
[Detectron2] (https://github.com/facebookresearch/detectron2)<br>
[YOLOx] (https://github.com/Megvii-BaseDetection/YOLOX)<br>
[DMMR] (https://github.com/boycehbz/DMMR)<br>
