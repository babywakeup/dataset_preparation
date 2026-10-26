# 数据准备测试

本目录集中存放数据准备、节点、API 和浏览器测试。业务模块、启动脚本、工作流和示例图片仍保留在上一级目录。

## 从 Diffusers 仓库根目录运行

```bash
PYTHONPATH="$PWD/examples/dataset_preparation${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -m unittest discover \
  -s examples/dataset_preparation/tests -p 'test_*.py' -v
```

`PYTHONPATH` 用于导入上一级的 `object_inpainting` 和 `object_inpainting_studio`。测试模块间的共享 fixture 仍由 unittest 的测试目录搜索路径解析。

若已经进入数据准备目录：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  ../../.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

若使用其他 Python 环境，替换解释器路径。不要再使用只搜索数据准备根目录的旧命令；应显式搜索本 tests 目录。

## 只运行一类测试

修改 `-p` 参数即可，例如 `-p 'test_flux_random_recolor_node.py'`。浏览器测试使用 Playwright Chromium，并在未指定 `STUDIO_TEST_URL` 时自行启动临时服务。

模型节点测试使用替代模型输出验证接口，不自动下载模型或执行真实换色质量评估。

## pytest

若环境已安装 pytest 及仓库测试所需依赖，也可以从仓库根目录运行：

```bash
.venv/bin/python -m pytest examples/dataset_preparation/tests -q
```

本目录的 conftest 为 pytest 补充业务模块导入路径；它不会为 unittest 自动配置路径，因此 unittest 仍需使用上面的 `PYTHONPATH`。
