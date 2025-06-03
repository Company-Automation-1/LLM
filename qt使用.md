# Qt Designer 使用指南

## 🛠️ 启动 Qt Designer

### PyQt5 用户

1. 安装工具包：
   ```bash
   pip install pyqt5-tools
   ```
2. 启动 Designer：
   - 直接运行：`python安装目录\Lib\site-packages\qt5_applications\Qt\bin\designer.exe`
   - **推荐** 添加环境变量后，终端直接运行：`designer`

### PySide6 用户

1. 安装 PySide6：
   ```bash
   pip install pyside6
   ```
2. 启动 Designer：
   ```bash
   pyside6-designer
   ```

## 🔁 UI 文件转 Python 代码

### PyQt5 转换命令

```bash
pyuic5 输入文件.ui -o 输出文件.py
```

示例：

```bash
pyuic5 mainwindow.ui -o ui_mainwindow.py
```

### PySide6 转换命令

```bash
pyside6-uic 输入文件.ui -o 输出文件.py
```

或使用重定向：

```bash
pyside6-uic mainwindow.ui > ui_mainwindow.py
```

## 💡 最佳实践建议

1. **文件命名规范**：

   - 保持 `.ui` 和生成的 `.py` 文件同名（如 `mainwindow.ui` → `ui_mainwindow.py`）
   - 添加 `ui_` 前缀区分原始界面文件

2. **路径处理**：

   ```bash
   # 处理子目录文件
   pyuic5 ./src/interface.ui -o ./ui/compiled_interface.py
   ```

3. **实时预览**：

   - 在 Designer 中按 `Ctrl+R` 快速预览界面
   - 修改后保存 `.ui` 文件，重新运行转换命令

4. **版本兼容**：
   ```bash
   # PySide2 旧版本
   pyside2-uic mainwindow.ui -o ui_mainwindow.py
   ```

> **提示**：转换后的 `.py` 文件不要手动编辑！所有界面修改应在 Designer 中完成，重新生成代码
