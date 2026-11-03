# 自定义物品训练接口

当前 Ubuntu 环境使用 Ultralytics `8.3.168`。本模块支持目标检测 `detect` 和实例分割
`segment` 训练；现有三维框、XYZ 和 Rx/Ry/Rz 仍由 D435i 深度点云计算，不需要标注三维框。

## 1. 采集与标注

建议每类先准备 300–1000 张真实工作场景图像，覆盖远近、方向、遮挡、亮暗、桌面背景和
机械臂阴影，并加入不含目标的负样本。按约 8:2 划分训练集和验证集。同一段视频的连续帧
不要随机分到两边，否则验证分数会虚高。

GUI 已内置检测框标注工具，不需要人工计算中心点或宽高：

1. 进入“YOLO 模型 → 训练新的物品模型”，点击“可视化标注图片”。
2. 在顶部选择 `train` 或 `val`，点击“导入图片”；也可以先启动相机后点击
   “加入当前相机帧”。
3. 在右侧选择物品类别，在图片上按住鼠标左键拖出矩形框，松开后自动保存。
4. 画错时在右侧选择对应框并删除。上一张、下一张会直接读取已保存标签。
5. 点击“管理类别”可在末尾增加类别。已有标签后禁止改动原类别的名称和顺序，避免类别
   编号错位。
6. 标完后关闭标注窗口，点击“校验数据集”，通过后开始训练。

程序根据鼠标框的左上角和右下角自动计算标签。检测标签每行仍采用标准YOLO格式：

```text
class_id x_center y_center width height
```

类别从 0 开始；四个坐标均为相对图像宽高的 0–1 数值。内置工具只负责目标检测矩形框；
实例分割需要用 CVAT、Label Studio、Roboflow 等工具画多边形并导出YOLO分割格式。

目录模板已经放在 `datasets/custom`：

```text
datasets/custom/
├── data.yaml
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

修改 `data.yaml` 的类别，例如：

```yaml
path: .
train: images/train
val: images/val
names:
  0: red_part
  1: blue_part
```

## 2. 三种训练入口

GUI：进入“YOLO 模型 → 训练新的物品模型”，选择 `data.yaml`，先点“校验数据集”，再设置
轮数、图像尺寸、批次和设备并开始训练。训练时相机会自动停止以释放 CPU/内存。完成后
`best.pt` 会自动填入模型路径框，点击“加载/热切换模型”即可使用。

命令行：

```bash
cd /home/user/d435i_camera_module
source .venv/bin/activate

# 只检查数据集
d435i-yolo-train --data datasets/custom/data.yaml --validate-only

# 检测模型训练
d435i-yolo-train \
  --data datasets/custom/data.yaml \
  --model models/yolov8n.pt \
  --task detect --epochs 100 --imgsz 640 --batch 8 --device cpu \
  --name custom_object
```

Python：

```python
from d435i_camera.training import TrainingConfig, YoloTrainingService

service = YoloTrainingService("/home/user/d435i_camera_module")
request = TrainingConfig(
    dataset_yaml="datasets/custom/data.yaml",
    base_model="models/yolov8n.pt",
    task="detect",
    epochs=100,
    image_size=640,
    batch=8,
    device="cpu",             # NVIDIA GPU 使用 "0"
    name="custom_object",
)
summary = service.validate(request)
result = service.train(request, progress=print)
print(result.best_model)
```

结果默认写入 `training_runs/<任务名称>/weights/best.pt` 和 `last.pt`。

## 3. 重要约定

- Ubuntu 主机当前没有可用 CUDA GPU，CPU 可以训练但会很慢；正式数据集建议放到 NVIDIA
  GPU 机器训练，再把 `best.pt` 复制回 `models/`。
- 只用“新物品”数据微调会让模型偏向新类别。如果同一个模型还必须保留人、水瓶、杯子、
  手机等 COCO 类别，训练集必须同时包含并标注这些类别，或由接入方运行两个模型后合并结果。
- 新类别名称会来自训练数据的 `names`。需要中文显示时可以直接在 `names` 中使用中文，或在
  `src/d435i_camera/labels.py` 增加英文到中文映射。
- `segment` 任务需要分割标签和 `yolov8n-seg.pt` 基础模型；图像平面 OBB 不是三维框。
