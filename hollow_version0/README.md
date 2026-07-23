# 空心粉推理封装

这个目录是空心粉检测的独立推理包，用于网页或其他程序调用。输入可以是一张图片，也可以是一个包含图片的 zip 文件；输出沿用 `give/back` 结构。

## 快速启动

```bash
cd /root/code/empty_heart/ans
python3 -m pip install -r requirements.txt
python3 code/web_app.py --host 0.0.0.0 --port 7860
```

浏览器打开：

```text
http://<服务器IP>:7860
```

如果服务器端口没有开放，可以在本地建立 SSH 隧道：

```bash
ssh -p 22037 -L 7860:127.0.0.1:7860 root@115.190.90.101
```

然后打开：

```text
http://127.0.0.1:7860
```

## 目录结构

```text
ans/
  code/                 推理代码和简易网页
    run_report.py       命令行入口
    web_app.py          网页入口
    mytools/            总体流程配置和封装
    eval_particle/      particle 推理逻辑
    hollow_pipeline/    hollow 面积比例过滤逻辑
  model/
    particle/particle.pt
    hollow/hollow.pt
  data/
    give/               输入图片目录
    back/               输出结果目录
    uploads/            网页上传的临时文件
```

## 命令行使用

单张图片：

```bash
cd /root/code/empty_heart/ans
python3 code/run_report.py --input /path/to/image.png --name demo --reset
```

图片文件夹：

```bash
cd /root/code/empty_heart/ans
python3 code/run_report.py --input /path/to/folder --name demo_folder --reset
```

只推理 hollow：

```bash
cd /root/code/empty_heart/ans
python3 code/run_report.py --input /path/to/folder --name demo_hollow --hollow-only --reset
```

## 输出内容

结果会写入：

```text
data/back/<任务名>/
  particle/       单独展示 particle 圈选结果
  hollow/         单独展示 hollow 候选和面积比例
  pic/            原有综合图
  pic_final/      最终汇报图：particle 统一蓝色，空心粉标注比例
  work/           中间文件和候选明细
  summary.json    结构化统计结果
  summary.txt     文本统计结果
```

网页会按同一张图片横向展示三列：`pic_final`、`hollow`、`particle`，方便直接对比最终结果和两个模型的单独输出。

## 默认参数

particle 使用当前记录的 first6 模型，hollow 使用最新的 hollow 联合训练模型。主要推理参数统一写在：

```text
code/mytools/config.py
```

后续更换权重、调整小图数量、置信度或空心率阈值时，优先改这个配置文件，避免在多个脚本里散落硬编码。
