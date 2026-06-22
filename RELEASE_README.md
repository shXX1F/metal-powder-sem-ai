# Metal Powder SEM GUI 便携版使用说明

## 给使用者

1. 解压 `MetalPowderSEM_GUI.zip`。
2. 双击 `start_gui.bat`。
3. 浏览器会自动打开本地页面：`http://127.0.0.1:8501`。
4. 上传 SEM 图像，输入比例尺参数，然后点击界面中的识别按钮。
5. 分析结果会保存在软件目录下的 `runs/gui/时间戳/` 文件夹中。

注意：请不要删除 `.runtime`、`metal_powder_sem_ai`、`app_streamlit.py` 等文件，否则软件无法启动。

## 给开发者

在项目根目录执行：

```powershell
.\build_release.ps1
```

脚本会生成：

```text
dist/
  MetalPowderSEM_GUI/
  MetalPowderSEM_GUI.zip
```

把 `dist/MetalPowderSEM_GUI.zip` 发给用户即可。用户不需要安装 Python，也不需要运行代码。

如果构建时下载依赖很慢，可以先配置国内 pip 镜像，例如：

```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

然后重新执行：

```powershell
.\build_release.ps1
```
