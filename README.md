# 图像分类任务模板

使用torchnet模板和tensorboard可视化分类模板。

models文件夹下为模型，对于自定义的模型，需要将模型写入models/\_\_init\_\_.py文件中才能在main.py中使用。

## Usage
clone该Repository
```
git clone git@github.com:JunLab-Hitsz/Classification-Template.git
cd Classification-Template
```

### Install Requirements
```
pip install -r requirements.txt
```

### Start Training
直接使用launch.sh脚本启动训练，传入gpu编号即可
```
$ ./launch.sh 0
```

或者使用如下命令，如下命令可以修改main.py中的parser参数
```
$ CUDA_VISIBLE_DEVICES=0 python main.py --dataset mnist
```

### Tensorboard
在命令行下可以直接运行以下代码启动Tensorboard
```
tensorboard --logdir=/path/to/tensorboard/log
```
修改以上logdir，该Repository的logdir为当前文件夹。

如果在docker中训练，由于端口可能无法看到tensorboard的结果，这里推荐使用VSCode来启动tensorboard

#### VSCode with Tensorboard
由于代码中存在Tensorboard的代码，VSCode会提示安装Tensorboard，安装完成后使用`Shift+Ctrl+p`打开VSCode菜单，输入`launch tensorboard`，随后选择logdir即可。此时VSCode下方控制台port栏会出现端口映射，鼠标点击即可访问tensorboard可视化结果。
![图片](https://user-images.githubusercontent.com/71539436/139179486-a686323d-392a-4c21-b421-5f29b56d3457.png)

## Reference
https://github.com/gram-ai/capsule-networks


## TODO
当前版本为单卡训练版本，多卡还没实现。

目标检测任务可以使用MMdetection框架，该框架基于Pytorch实现。
