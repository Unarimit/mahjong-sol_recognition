# mahjong-sol_recognition
识别雀魂麻将桌上歪曲扭八的麻将牌，目标是实现一个自动避铳机。现在还停留在目标识别阶段

部分数据来源于：https://github.com/MaZhengKe/mahjong

Prototypical Networks实现参考：https://github.com/oscarknagg/few-shot 

## Run
运行`ImgProcessor.py`中的主函数即可

## 现状
完成了传统视觉的识别方式，准确度80%

<img width="500" src="docs/result-preview.png"/>

使用ProtoNet网络对麻将牌进行分类，实际表现...感觉不太行

<img width="500" src="docs/result-protonet.png"/>

原来是cv2读图片颜色通道是BGR的，我训练时dataloader读图片颜色通道是RGB的。改了一下感觉还行，主要问题还是在万字牌上

<img width="500" src="docs/result-protonet-update.png"/>

增加对万字牌重复识别的模型。尽管测试集上准确度0.92，但对万字牌的识别还是不太准

<img width="500" src="docs/result-protonet-19w.png"/>

## 预计
1. 添加使用神经网络的识别方式
    - 数据集制作 -> fin 每个牌种至少4张 
    - 已完成模型训练 -> fin 测试集上准确率97%
    - 整合进程序 -> fin 效果不理想，可能原因为输入像素太小（28\*28）难以识别万字牌的数字
    - 增大输入像素面积 -> fin 效果不理想，40\*40也难以识别万字牌的数字
    - 重新训练模型或改进网络
2. 自动化
