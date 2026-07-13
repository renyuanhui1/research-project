# Git + GitHub 代码管理指南

## 一、初始配置(只需一次)

检查是否已安装:
```bash
git --version
```

设置身份信息(会记录在每次提交里):
```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

## 二、创建/关联仓库

**新项目:**
```bash
cd 你的项目文件夹
git init
```

**已有GitHub仓库,拉到本地:**
```bash
git clone 仓库地址
```

## 三、.gitignore(重要)

项目根目录新建 `.gitignore` 文件,写上不想被追踪的内容,例如:
```
__pycache__/
.env
node_modules/
*.log
```

## 四、日常提交流程

```bash
git status              # 查看哪些文件被改动
git add .                # 把改动加入暂存区(也可 git add 文件名 只加单个文件)
git commit -m "说明这次改了什么"   # 正式记录一个版本
```

## 五、连接并推送到GitHub

1. 在GitHub网页新建一个空仓库(New repository),复制仓库地址
2. 本地关联(只需一次):
```bash
git remote add origin 仓库地址
```
3. 推送:
```bash
git push -u origin main   # 第一次加 -u,之后直接 git push 即可
```
> 如果远程仓库不是空的(比如已有README),先 `git pull` 合并再push。

## 六、日常循环

改文件 → `git add .` → `git commit -m "说明"` → `git push`

换电脑或多人协作时,先 `git pull` 拉取最新改动。

## 七、分支管理

```bash
git checkout -b 新功能名    # 创建并切换到新分支
# ...改代码、commit...
git checkout main           # 切回主分支
git merge 新功能名          # 确认没问题后合并进main
```

## 八、查看历史 & 撤销

```bash
git log --oneline              # 简洁查看提交历史
git checkout -- 文件名          # 撤销某个文件未commit的修改
git commit --amend             # 修改最近一次的提交信息
git revert 提交哈希             # 已push的提交,安全撤销用revert(别用reset)
```

## 九、关于登录认证

GitHub推送现在不支持直接用密码,两种方式二选一:

- **个人访问令牌(Personal Access Token)**:GitHub设置里生成一个token,push时密码栏输入token
- **SSH Key**:配置一次以后无需每次输入密码,更省事

## 十、常用命令速查表

| 命令 | 作用 |
|---|---|
| `git status` | 查看当前改动状态 |
| `git add .` | 添加所有改动到暂存区 |
| `git commit -m "xxx"` | 提交改动 |
| `git push` | 推送到GitHub |
| `git pull` | 拉取远程最新改动 |
| `git log --oneline` | 查看提交历史 |
| `git checkout -b 分支名` | 创建并切换新分支 |
| `git merge 分支名` | 合并分支 |
| `git clone 地址` | 克隆远程仓库到本地 |
