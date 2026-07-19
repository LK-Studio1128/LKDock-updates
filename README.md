# LKDock-updates

LKDock 在线更新公共仓库。

## 目录结构

```
v1.0/update_manifest.json   # v1.0 更新清单
v2.0/update_manifest.json   # v2.0 更新清单
v3.0/update_manifest.json   # v3.0 更新清单
v4.0/update_manifest.json   # v4.0 更新清单
```

## 客户端访问地址

```
https://raw.githubusercontent.com/LK-Studio1128/LKDock-updates/main/v1.0/update_manifest.json
https://raw.githubusercontent.com/LK-Studio1128/LKDock-updates/main/v2.0/update_manifest.json
https://raw.githubusercontent.com/LK-Studio1128/LKDock-updates/main/v3.0/update_manifest.json
https://raw.githubusercontent.com/LK-Studio1128/LKDock-updates/main/v4.0/update_manifest.json
```

## 更新包下载

更新包通过 GitHub Releases 发布：

```
https://github.com/LK-Studio1128/LKDock-updates/releases
```

## 当前版本

| 版本 | Release Tag | Win | Mac | Linux |
|------|-------------|:---:|:---:|:-----:|
| v1.0 | v1.0.3 | ✅ | ✅ | ❌ |
| v2.0 | v2.0.3 | ✅ | ✅ | ❌ |
| v3.0 | v3.0.3 | ✅ | ✅ | ❌ |
| v4.0 | v4.0.3 | ✅ | ✅ | ❌ |

## 架构说明

- **LKDock1-4** (源码仓库): CI 编译 main*.py → .so/.pyd → 生成增量 ZIP + SHA256
- **LKDock-updates** (本仓库): Release 上传 ZIP + manifest 回写 SHA256 → 客户端下载

## 发布流程

1. 修改 `打包/更新/versions.json` 中的目标版本号
2. 运行 `python3 bump_version.py` 同步版本号到所有源文件
3. 各平台编译 Nuitka 产物（.pyd / .so）
4. 打包为 zip 放到 `release_zips/` 目录
5. 运行 `python3 release_and_upload.py v4.0 4.0.4 "更新说明"`

详见源码仓库中的 `打包/更新/发布指南.md`。
