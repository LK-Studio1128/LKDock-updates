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

## 架构说明

- **LKDock1-4** (源码仓库): CI 编译 main*.py → .so/.pyd → 生成增量 ZIP + SHA256
- **LKDock-updates** (本仓库): Release 上传 ZIP + manifest 回写 SHA256 → 客户端下载
