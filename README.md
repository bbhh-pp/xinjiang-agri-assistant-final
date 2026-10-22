# 新疆农牧小助手（整合最终版）

这是可在本机运行的单文件网页应用。`app.py` 内含 Python 后端、网页界面与农牧词表；启动后用浏览器访问，不需要先部署网站。本目录只保留运行与评审所需文件，不包含 API 密钥、用户数据或语音模型。

## 评委快速运行

1. 安装 Python 3.10 或更新版本，并确认终端中的 `python --version` 可用。
2. 在**本目录**打开终端，先运行离线检查：
   ```powershell
   python app.py --self-check
   ```
   这一步不调用神农、腾讯或讯飞接口，可用于确认程序和网页资源完整。
3. 如需体验真实问答，在**启动程序的同一个 PowerShell 窗口**中输入评委自己申请的密钥：
   ```powershell
   $env:SHENNONG_API_KEY = Read-Host "神农 API Key"
   $env:TENCENT_TOKENHUB_API_KEY = Read-Host "腾讯 TokenHub API Key"
   python app.py
   ```
4. 在这台电脑的浏览器打开 <http://127.0.0.1:8765/>。使用结束后，在启动窗口按 `Ctrl+C`。

没有密钥也能启动并查看界面，但真实 AI 问答、翻译和图片分析不能完成。上述密钥只保存在当前终端进程中；关闭窗口后需重新设置。**不要把真实密钥写进代码、截图、提交记录或公开仓库。** 接口调用可能产生费用，以各服务提供方的账户设置为准。

## 功能与所需配置

| 功能                 | 需要的配置                                                                               |
| ------------------ | ----------------------------------------------------------------------------------- |
| 查看页面、离线自检          | Python 3.10+；无需 API 密钥                                                              |
| 中文文字问答、图片提问、中维双语回答 | `SHENNONG_API_KEY` 与 `TENCENT_TOKENHUB_API_KEY`；需能访问对应在线服务                          |
| 维吾尔语文字提问           | 上述两个密钥；翻译由腾讯 TokenHub 完成，回答由神农生成                                                    |
| 中文语音输入、中文朗读        | 依赖浏览器提供相应语音功能，并允许麦克风访问；可用性和声音效果随浏览器及系统变化                                            |
| 维吾尔语语音输入           | 上述两个密钥，另需下述本地 ONNX 依赖、语音模型和 FFmpeg                                                  |
| 维吾尔语朗读             | 另需讯飞语音合成的 `XFYUN_TTS_APP_ID`、`XFYUN_TTS_API_KEY`、`XFYUN_TTS_API_SECRET`，且账户已开通所用发音人 |

需要维吾尔语朗读时，在启动前于同一个 PowerShell 窗口设置讯飞凭据：

```powershell
$env:XFYUN_TTS_APP_ID = Read-Host "讯飞 APPID"
$env:XFYUN_TTS_API_KEY = Read-Host "讯飞 APIKey"
$env:XFYUN_TTS_API_SECRET = Read-Host "讯飞 APISecret"
```

## 可选依赖：维语语音识别

基础网页仅使用 Python 标准库，不必安装 `requirements.txt`。只有要体验**维语语音输入**时，才运行：

```powershell
python -m pip install -r requirements.txt
python app.py --download-model
```

`requirements.txt` 包含 ONNX Runtime、NumPy 和 `imageio-ffmpeg`。如果系统已安装 FFmpeg，程序也可直接使用系统中的 FFmpeg。模型下载命令会从上游项目下载 ONNX 文件并校验 SHA-256；模型默认存放在用户缓存目录，**不会放进本仓库**。下载需要网络连接。

模型来自 [gheyret/uyghurasr\_python](https://github.com/gheyret/uyghurasr_python) 的 `tunji` 发布文件。本项目的 MIT 许可证**不授予第三方模型的使用或再分发权**；公开部署、再分发或商业使用该模型前，应向模型作者确认授权。

## 数据、安全与发布说明

- 问答历史保存在运行者电脑的本地数据目录：Windows 默认位于 `%LOCALAPPDATA%\XinjiangPastoralAssistant\data`，其他系统默认使用用户数据目录。该目录不会随本仓库一起发布。
- 当前版本**免登录，且只监听本机地址** `127.0.0.1`。不要直接把它暴露到公网；多人在线试用需要额外的身份验证、用量限制和数据隔离。
- 本项目代码采用 [MIT 许可证](LICENSE)。神农、腾讯、讯飞接口及第三方语音模型分别受各自的服务条款或授权约束。

## 目录内容

```text
整合最终版/
├─ app.py             程序与网页
├─ README.md          运行与评审说明
├─ requirements.txt   可选的维语语音识别依赖
└─ LICENSE            MIT 许可证
```
