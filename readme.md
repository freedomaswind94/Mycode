## 数据形式
    train:
        A
        B
        label
    vl:
        A
        B
        label
    test:用于预测
        A
        B
## label
    为0-1二值图，使用./corecode/tool/255lable2zeroandone.py转换
## 训练：
    1.\corecode\tool\levircdTotxt.py 配置数据路径（数据为裁剪后的），生成数据路径文件
    2.train.py   
## 预测：
    infer.py
## 补充：
    1.数据增强：
      在训练模式下使用albumentations对输入影像增强，可在\corecode\dataset\transforms.py
      下对albtrans 修改自定义；
      val 和 test模式下不做数据增强；
    2.数据裁切：
      levirCD单张为1024*1024，需要根据实际裁剪成相应大小的样本 \corecode\tool\preprocess_levir_cd.py
    3.损失函数使用混合损失处理类别失衡： FC_loss+Dice_loss 
