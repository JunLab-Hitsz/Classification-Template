# 图像分类任务模板

使用torchnet模板和tensorboard可视化分类模板。

## Install Requirements
```
pip install -r requirements.txt
```

models文件夹下为模型，对于自定义的模型，需要将模型写入models/__init__.py文件中才能在main.py中使用。

## Usage
可以直接使用launch.sh脚本启动训练，传入gpu编号即可
```
$ ./launch.sh 0
```

或者使用如下命令，如下命令可以修改main.py中的parser参数
```
$ CUDA_VISIBLE_DEVICES=0 python main.py --dataset mnist
```

## Reference：
https://github.com/gram-ai/capsule-networks


## TODO
当前版本为单卡训练版本，多卡还没实现。

目标检测任务可以使用MMdetection框架，该框架基于Pytorch实现。