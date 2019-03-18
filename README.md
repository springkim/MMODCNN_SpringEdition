MMODCNN_SpringEdition <img src="https://i.imgur.com/oYejfWp.png" title="Windows8" width="48">
--------------------------------------------------------------------------------------------
<img src="https://i.imgur.com/ElCyyzT.png" title="Windows8" width="48"><img src="https://i.imgur.com/O5bye0l.png" width="48"><img src="https://i.imgur.com/kmfOMZz.png" width="48"><img src="https://i.imgur.com/6OT8yM9.png" width="48">

Max-margin object detection convolution neural networks.

#### MMODCNN C++ Windows and Linux interface library. (Train,Detect both)

-	Remove dlib dependency.
-	You need only 1 files for MMODCNN object detection.
-	Support windows, linux as same interface.

#### Example detection code.

```cpp
MMODCNN detector;
detector.Create("vehicle.dat", "names.txt");
cv::Mat img=cv::imread("a.jpg");
std::vector<BoxSE> boxes = detector.Detect(img, 0.5F);
```

### 1. Setup for train.

#### 1.1. Train detector
You need only 2 files for train that are **MMODCNNSE_Train.exe** and **cudnn64_7.dll** on Windows. If you are on Linux, then you need only **MMODCNNSE_Train**. This files are in `build/Release` after run cmake build.


There is a example training directory `MMODCNN_SpringEdition_Train/`. You can start training using above files.

The **MMODCNN_Train.exe**'s arguments are `[directory]`,`[xml]`,`[network name]`,`[mini batch size]` , `[min detection width=40]`, `[min detection height=40]`, `[solver=sgd]`, `[Cropper=350]` and `[min learning rate=0.0001]`.

##### Example
```
MMODCNNSE_Train.exe . training.xml hello 32 40 40 adam 500
```

### 2. Setup for detect

Just include **MMODCNNSE.h** and use it. See `MMODCNN_SpringEdition_Test/`. You need only **MMODCNNSE.h**, **libMMODCNNSE.dll**, **opencv_world400.dll** and **cudnn64_7.dll** for detect.


##### Reference

The class `MMODCNN` that in `MMODCNNSE.h` has 3 methods.

```cpp
void Create(std::string dat, std::string names);
```

This method load trained model(**weights**) that has **dat** extension, and class naming file(**names**)

* **Parameter**
	* **dat** : trained model path(e.g. "vehicle.dat")
	* **names** : class naming file(e.g. "names.txt")

```cpp
std::vector<BoxSE> Detect(std::string file, float threshold);
std::vector<BoxSE> Detect(cv::Mat img, float threshold);
```

This method is detecting objects or classify of `file`,`cv::Mat` or `IplImage`.
* **Parameter** 
	* **file** : image file path 
	* **img** : 3-channel image. 
	* **threshold** : It removes predictive boxes if there score is less than threshold.

```cpp
void Release();
```
Release loaded network.



Technical issue
---------------

This is dlib's MMODCNN wrapper. I added dynamic mini-batch loading module when training one step.

change log
----------

**build_windows.bat** and **build_linux.sh** will download automatically correct version of cudnn. and build as cmake.

```
Windows + 1080ti + CUDA 10.0 + cudnn7.5      = 28FPS
```

Software requirement
--------------------

-	CMake
-	CUDA 10.0
-	OpenCV(for testing)(included in repository)
-	Visual Studio 2013, 2015, 2017

Hardware requirement
--------------------

-	NVIDIA GPU