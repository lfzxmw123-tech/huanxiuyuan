# GAMES101 学习路线：用 Python / JavaScript 实现

这份文档帮助你在不配置 C++、CMake 和 OpenCV 的情况下，学习 GAMES101 的核心内容。

## 选择哪种语言？

**学习介绍：** 这一部分用于明确 Python、JavaScript 和 Three.js 各自适合解决的问题。学习时不必在两种语言之间二选一，而应把 Python 当作算法验证工具，把 JavaScript 当作交互式可视化工具，从而用同一套数学知识完成离线计算和浏览器展示。

**练习代码：**

```python
import numpy as np
print("Python:", np.array([1, 2, 3]) * 2)
```

| 目标 | 推荐方案 |
| --- | --- |
| 学习向量、矩阵和图形学公式 | Python + NumPy |
| 在浏览器中实时看到图形 | JavaScript + Canvas |
| 快速验证完整 3D 效果 | JavaScript + Three.js |

建议先用 Python 理解算法，再用 JavaScript 做可视化。

## Python 环境准备

**学习介绍：** 这一部分帮助你搭建最小可用的图形学实验环境。目标不是一次安装大量框架，而是先掌握 NumPy 的数组与矩阵计算，以及 Pillow 的像素读写和图片输出，为后面的变换、光栅化和离线渲染打基础。

**练习代码：**

```python
import numpy as np
from PIL import Image

matrix = np.eye(4, dtype=np.float32)
Image.new("RGB", (32, 32), "black").save("test.png")
print(matrix)
```

安装 Python 后，在终端执行：

```bash
pip install numpy pillow
```

其中：

- `numpy` 用于向量和矩阵计算
- `pillow` 用于创建、绘制和保存图片

## C++ 和 Python 的对应关系

**学习介绍：** 这一部分帮助你阅读 GAMES101 的 C++ 示例，并把其中的数据结构和绘制操作转换为 Python。重点是理解不同语言虽然语法不同，但向量、矩阵、图像缓冲区和像素写入等核心概念完全对应。

**练习代码：**

```python
import numpy as np
vertex = np.array([1, 2, 3], dtype=np.float32)  # Vector3f
image = np.zeros((100, 100, 3), dtype=np.uint8)  # cv::Mat
image[50, 50] = [255, 255, 255]               # set_pixel
```

| C++ 作业 | Python 写法 |
| --- | --- |
| `Eigen::Vector3f` | `numpy.ndarray` |
| `Eigen::Matrix4f` | `numpy.ndarray` |
| `cv::Mat` | NumPy 图像数组 |
| `rasterizer::draw()` | Python 的 `draw()` 函数 |
| `set_pixel()` | 图像数组下标赋值 |
| `cv::imwrite()` | `PIL.Image.save()` |

## Assignment1 的核心流程

**学习介绍：** 这一部分先建立完整的渲染管线概念：一个三维顶点如何依次经过模型、视图和投影变换，最终落到二维屏幕。先理解数据在各坐标空间之间如何流动，再编写代码，会更容易定位矩阵顺序和坐标方向问题。

**练习代码：**

```python
import numpy as np
vertex = np.array([1., 0., -2., 1.])
model = np.eye(4)
view = np.eye(4)
projection = np.eye(4)
print(projection @ view @ model @ vertex)
```

Assignment1 的任务是把一个 3D 三角形变换并绘制到 2D 图片上：

```text
三角形顶点
    ↓
模型变换 Model
    ↓
视图变换 View
    ↓
投影变换 Projection
    ↓
透视除法
    ↓
屏幕坐标
    ↓
画线或填充三角形
```

最终变换可以写成：

```text
屏幕坐标 = Projection × View × Model × 顶点
```

矩阵从右向左生效，所以顶点会先经过 Model，再经过 View，最后经过 Projection。

## 第一个 Python 示例：旋转并绘制三角形

**学习介绍：** 这一部分通过一个可直接运行的最小程序，把矩阵变换、齐次坐标、屏幕映射和图片绘制串联起来。运行示例后，建议修改旋转角度、顶点坐标和画布尺寸，观察各项参数怎样影响最终图像。

保存为 `assignment1_wireframe.py`：

```python
import numpy as np
from PIL import Image, ImageDraw

WIDTH = 700
HEIGHT = 700


def get_model_matrix(angle_degrees):
    """生成绕 Z 轴旋转的模型矩阵。"""
    theta = np.radians(angle_degrees)
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float32)


def ndc_to_screen(point):
    """把 [-1, 1] 范围的 NDC 坐标转换为像素坐标。"""
    x = int((point[0] + 1) * WIDTH / 2)
    y = int((1 - point[1]) * HEIGHT / 2)
    return x, y


vertices = np.array([
    [2, 0, -2, 1],
    [0, 2, -2, 1],
    [-2, 0, -2, 1],
], dtype=np.float32)

model = get_model_matrix(30)

# 矩阵乘以每个顶点。
transformed = (model @ vertices.T).T

# 齐次坐标透视除法。当前 w 都是 1，所以结果看起来不明显，
# 但后续加入投影矩阵后这一步非常重要。
transformed = transformed[:, :3] / transformed[:, 3, None]

points = [ndc_to_screen(vertex) for vertex in transformed]

image = Image.new("RGB", (WIDTH, HEIGHT), "black")
draw = ImageDraw.Draw(image)
draw.line([points[0], points[1], points[2], points[0]],
          fill="white", width=2)

image.save("output.png")
print("已生成 output.png")
```

运行：

```bash
python assignment1_wireframe.py
```

## 模型旋转矩阵

**学习介绍：** 这一部分单独讲解模型变换中最常用的旋转矩阵。学习重点是理解正弦、余弦如何改变顶点方向，以及角度制与弧度制的区别；掌握它之后，可以继续组合平移和缩放矩阵。

**练习代码：**

```python
import numpy as np
theta = np.radians(90)
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
print(rotation @ np.array([1., 0.]))
```

绕 Z 轴旋转角度 `θ` 的矩阵是：

```text
[ cosθ  -sinθ   0   0 ]
[ sinθ   cosθ   0   0 ]
[  0      0     1   0 ]
[  0      0     0   1 ]
```

Python 中角度默认需要转换成弧度：

```python
theta = np.radians(angle_degrees)
```

## JavaScript Canvas 版本

**学习介绍：** 这一部分把绘制结果搬到浏览器中，帮助你熟悉 Canvas 的坐标系和基本绘图 API。当前示例直接使用屏幕坐标，后续可以逐步加入与 Python 相同的矩阵运算，并用动画和输入控件实现实时交互。

```javascript
const ctx = document.querySelector("canvas").getContext("2d");
ctx.fillStyle = "red"; ctx.fillRect(10, 10, 20, 20);
```

保存为 `assignment1_canvas.html`，然后用浏览器打开：

```html
<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>Assignment1 线框三角形</title>
<canvas id="canvas" width="700" height="700"></canvas>
<script>
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.beginPath();
ctx.moveTo(350, 100);
ctx.lineTo(500, 400);
ctx.lineTo(200, 400);
ctx.closePath();

ctx.strokeStyle = "white";
ctx.lineWidth = 2;
ctx.stroke();
</script>
```

Canvas 版本先直接使用屏幕坐标，后续可以把 Python 里的矩阵变换也移植到 JavaScript 中。

## 推荐学习顺序

**学习介绍：** 这一部分给出从数学基础到路径追踪的渐进路线。建议按顺序学习并为每个阶段保留可运行结果，因为后续作业会反复复用前面实现的向量、矩阵、求交和采样功能。

```python
for assignment in range(9): print(f"开始 Assignment{assignment}")
```

### Assignment0：Eigen 和矩阵基础

**学习介绍：** 本阶段负责补齐图形学所需的线性代数基础。你需要能够用代码表示向量和矩阵，并理解点积、叉积与齐次坐标的几何意义，而不只是记住公式。

```python
import numpy as np; print(np.dot([1,2,3], [3,2,1]))
```

用 NumPy 学会：

- 向量加法和数乘
- 矩阵乘法
- 点积和叉积
- 齐次坐标
- 平移、旋转、缩放矩阵

### Assignment1：变换和光栅化

**学习介绍：** 本阶段学习把三维物体变换到二维屏幕，并完成最基础的线框与三角形光栅化。它连接了线性代数和实际像素，是理解实时渲染管线的关键一步。

```python
import numpy as np; print((np.eye(4) @ np.array([1,0,0,1]))[:2])
```

依次实现：

1. Model 矩阵
2. View 矩阵
3. Projection 矩阵
4. 透视除法
5. 屏幕坐标映射
6. Bresenham 画线
7. 三角形包围盒
8. 重心坐标判断点是否在三角形内

### Assignment2：三角形填充和深度测试

**学习介绍：** 本阶段从“画出边框”进入“生成每个像素”。你将使用重心坐标判断采样点是否位于三角形内，并通过深度缓冲解决多个三角形之间的前后遮挡关系。

```python
inside = lambda a,b,c: a >= 0 and b >= 0 and a+b <= 1; print(inside(.2,.3,.5))
```

核心公式是重心坐标：

```text
P = αA + βB + γC
α + β + γ = 1
```

如果 `α、β、γ` 都大于等于 0，点 `P` 就在三角形内部。

### Assignment3：Phong 光照

**学习介绍：** 本阶段学习如何根据光源、表面法线、观察方向和材质计算像素颜色。重点是区分环境光、漫反射和镜面反射的作用，并理解顶点属性如何插值到三角形内部。

```python
import numpy as np; print(max(np.dot([0,0,1], [0,0,1]), 0.0))
```

学习：

- 环境光
- 漫反射
- 镜面反射
- 法向量
- 光照插值

### Assignment4：Bezier 曲线

**学习介绍：** 本阶段从三角形渲染转向曲线生成，通过反复线性插值理解 De Casteljau 算法。学习后应能说明控制点如何影响曲线形状，并实现平滑采样与显示。

```python
p=[(0,0),(1,2),(2,0)]; t=.5; print((1-t)**2*p[0][1] + 2*(1-t)*t*p[1][1] + t*t*p[2][1])
```

学习递归插值和 De Casteljau 算法。

### Assignment5～7：光线追踪与路径追踪

**学习介绍：** 这一阶段学习从相机发射光线来生成图像，并逐步加入空间加速、随机采样和全局光照。它与光栅化采用不同的成像思路，能够帮助你理解反射、阴影、间接光照和渲染噪声的来源。

```python
import numpy as np; ray=np.array([0.,0.,-1.]); print(ray/np.linalg.norm(ray))
```

学习：

- 光线与物体求交
- 递归反射
- 阴影
- 蒙特卡洛采样
- 路径追踪

## 常见问题

**学习介绍：** 这一部分汇总初学实现中最容易遇到的路线选择、结果差异和运行方式问题。调试时应先检查数学和坐标约定，再检查绘图库或运行环境，避免把算法错误误判为工具问题。

```python
assert abs(1.0 - 1.0) < 1e-6
```

### 为什么不直接使用 Three.js？

**学习介绍：** 这个问题帮助你区分“学习底层渲染原理”和“快速制作三维效果”两种目标。理解 Three.js 替你封装了哪些步骤之后，才能在合适的阶段使用它，而不会跳过关键知识。

```javascript
const canvas = document.querySelector("canvas");
console.log(canvas ? "Canvas 已就绪" : "请先创建 canvas");
```

Three.js 会替你完成很多底层工作，适合做效果验证，但不适合一开始理解光栅化器。建议先手写 Python 版本，再用 Three.js 对照结果。

### Python 版本和 C++ 作业结果不一样怎么办？

**学习介绍：** 这个问题训练系统化调试能力。不同语言实现出现差异时，应使用相同输入并逐阶段对比中间结果，从矩阵、齐次坐标、NDC 到像素坐标逐层缩小问题范围。

```python
import numpy as np
np.testing.assert_allclose(np.eye(2) @ [1, 0], [1, 0])
```

优先检查：

1. 矩阵乘法顺序
2. 角度和弧度是否混用
3. 相机坐标系方向
4. 透视除法是否执行
5. 屏幕 Y 坐标是否需要翻转

### 如何运行带窗口的版本？

**学习介绍：** 这个问题说明离线图片与实时窗口之间的学习顺序。先确保单帧输出正确，再加入窗口、键盘输入和动画循环，可以把数学问题与交互框架问题分开处理。

```python
from PIL import Image
Image.new("RGB", (64, 64), "black").save("output.png")
```

先用 Pillow 生成 `output.png`。等矩阵和光栅化逻辑正确后，再换成 Pygame 做实时交互。

## 最小目标

**学习介绍：** 这一部分把 Assignment1 压缩成一个明确、可验证的小里程碑。先完成从输入角度到输出图片的闭环，再逐项增加填充、深度测试和交互功能，可以降低一次实现整套渲染器的难度。

```python
angle = 30
print(f"rendering triangle at {angle} degrees")
```

学习 Assignment1 时，先达到下面这个目标：

```text
输入一个旋转角度
    ↓
计算 Model / View / Projection
    ↓
把 3D 三角形投影到 2D
    ↓
在 output.png 中显示线框三角形
```

完成这个目标后，再加入三角形填充和深度测试。

## 完整课程实现范围

**学习介绍：** 这一部分展示完成整套课程时的项目边界和目录组织方式。拆分公共模块与各次作业，可以减少重复代码，并让每个知识点都有独立入口、测试数据和输出结果。

```python
from pathlib import Path
root = Path("games101-python")
for name in ["common", "assignment0", "assignment1", "assignment2"]:
    (root / name).mkdir(parents=True, exist_ok=True)
```

如果要把这套课程完整实现，建议把每个作业都转换成独立模块。

```text
games101-python/
├─ common/              # 向量、矩阵、图片和相机
├─ assignment0/         # 数学基础
├─ assignment1/         # 变换和线框光栅化
├─ assignment2/         # 三角形填充和深度测试
├─ assignment3/         # Phong 光照
├─ assignment4/         # Bezier 曲线
├─ assignment5/         # 基础光线追踪
├─ assignment6/         # BVH 加速结构
├─ assignment7/         # 路径追踪
└─ assignment8/         # 质点弹簧系统
```

### Assignment0：数学基础

**学习介绍：** 这一模块建立整个项目共用的数学工具层。除了实现运算，还应通过简单测试验证单位矩阵、逆矩阵和复合变换是否正确，确保后续渲染错误不会来自基础代码。

```python
import numpy as np; print(np.linalg.inv(np.eye(4)))
```

实现三维向量、四维齐次坐标、矩阵乘法、逆矩阵、平移、旋转、缩放、点积和叉积。

产出：一个可以打印向量、矩阵和变换结果的 Python 程序。

### Assignment1：变换和线框光栅化

**学习介绍：** 这一模块完整实现三维顶点到二维线框图像的过程。学习时要特别关注不同坐标空间、矩阵相乘顺序、透视除法和屏幕 Y 轴方向，这些约定将贯穿后续所有渲染任务。

```python
import numpy as np; p=np.array([1.,2.,3.,1.]); print((np.eye(4)@p)[:3])
```

实现 Model、View、Projection 矩阵、透视除法、NDC 到屏幕坐标的映射和 Bresenham 画线。

产出：旋转三角形图片或 Canvas 动画。

### Assignment2：三角形光栅化

**学习介绍：** 这一模块实现软件光栅化器的核心循环：遍历候选像素、判断内部、比较深度并写入颜色。它能让你直观看到 GPU 光栅化阶段在底层完成了哪些工作。

```python
depth=np.full((64,64), np.inf); depth[10,10]=.5; print(depth[10,10])
```

实现三角形包围盒、重心坐标、内部测试、深度缓冲区和颜色插值。

产出：填充的彩色三角形和正确的遮挡关系。

### Assignment3：Phong 光照

**学习介绍：** 这一模块为光栅化结果加入材质与光照，使几何形状呈现立体感。你将把位置、颜色、法线等属性插值到像素，并在片元级别计算 Phong 或 Blinn-Phong 光照。

```python
color = 0.1 + 0.8 * max(0.0, 1.0); print(color)
```

实现环境光、漫反射、镜面反射、法向量插值和 Phong 光照模型。

产出：带明暗变化的 3D 模型。

### Assignment4：Bezier 曲线

**学习介绍：** 这一模块通过控制点和递归插值生成连续曲线。除正确绘制曲线外，还要观察采样间隔对平滑度的影响，并尝试用多像素覆盖或颜色混合改善锯齿。

```python
points=[(i/10, i/10) for i in range(11)]; print(points[-1])
```

实现 De Casteljau 算法、递归插值、曲线采样和抗锯齿。

产出：可调整控制点的曲线绘制器。

### Assignment5：基础光线追踪

**学习介绍：** 这一模块建立光线追踪器的最小框架，包括相机射线、物体求交、最近交点和递归反射。完成后，你应能解释一条像素射线如何决定最终颜色，以及阴影射线如何判断光源可见性。

```python
origin, direction = (0,0,0), (0,0,-1); hit = False; print(origin, direction, hit)
```

实现光线与三角形或球体求交、可见性判断、阴影光线和反射递归。

产出：支持球体和三角形的离线渲染器。

### Assignment6：加速结构

**学习介绍：** 这一模块解决逐个测试所有物体导致的性能问题。通过 AABB 和 BVH，你将学习如何用空间层次结构跳过大量不可能相交的几何体，并用计时数据验证优化效果。

```python
box_min, box_max = (-1,-1,-1), (1,1,1); print(box_min, box_max)
```

实现 Axis-Aligned Bounding Box、BVH 节点、场景划分和光线与包围盒求交。

产出：对比使用 BVH 前后的渲染时间。

### Assignment7：路径追踪

**学习介绍：** 这一模块把直接光照扩展为包含多次反弹的全局光照。重点是理解概率采样、蒙特卡洛估计、俄罗斯轮盘赌和无偏性，并通过增加每像素采样数观察噪声逐渐收敛。

```python
import random; samples=[random.random() for _ in range(1000)]; print(sum(samples)/len(samples))
```

实现随机采样、余弦加权半球采样、蒙特卡洛积分、俄罗斯轮盘赌和多次反弹。

产出：支持间接光照的路径追踪图片。

### Assignment8：质点弹簧系统

**学习介绍：** 这一模块进入物理模拟，用质点和弹簧近似绳子、布料或软体。你需要把受力、速度和位置随时间更新，并比较不同积分方法、时间步长与阻尼对稳定性的影响。

```python
position, velocity, dt = 0.0, 0.0, 0.01; velocity += -9.8*dt; position += velocity*dt; print(position)
```

实现质点、弹簧、胡克定律、重力、积分、阻尼和碰撞。

产出：布料、绳子或软体模拟。

## Python 和 JavaScript 的分工

**学习介绍：** 这一部分说明同一项目如何在两种语言之间分配任务和复用知识。关键是统一坐标系、矩阵约定和输入数据，让 Python 的离线结果能够作为 JavaScript 实时实现的对照基准。

```javascript
const vertex = [1, 2, 3, 1];
console.log(vertex.map(Number));
```

| 模块 | Python | JavaScript |
| --- | --- | --- |
| 数学验证 | NumPy | 原生数组或 Three.js `Matrix4` |
| 离线渲染 | Pillow | Canvas 导出图片 |
| 实时交互 | Pygame | Canvas / WebGL |
| 3D 对照实验 | Open3D 或自定义渲染 | Three.js |

Python 版本负责验证算法，JavaScript 版本负责实时显示结果。两者使用相同的输入数据和数学公式。

## 实现顺序

**学习介绍：** 这一部分把知识依赖转换为实际开发顺序。每完成一个阶段，都应保存源码、运行命令、输入数据和期望图片；这样既能形成作品集，也方便在后续模块出现问题时回溯验证。

```python
from pathlib import Path
for n in range(9): Path(f"assignment{n}").mkdir(exist_ok=True)
```

```text
Assignment0 → Assignment1 → Assignment2 → Assignment3
                                      ↓
Assignment4 → Assignment5 → Assignment6 → Assignment7
                                                ↓
                                           Assignment8
```

每完成一个作业，都保留一个可独立运行的示例和一张输出图片，方便定位数学或坐标变换错误。
