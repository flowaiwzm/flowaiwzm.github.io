---
title: 'Git的常见命令'
date: 2025-4-23
tags:
  - cool posts
  - category1
  - category2
---
### Git的常见命令

- 项目初始化---- **git init**
- 查看当前管理中状态--- **git status**
  - **红色**：修改、创建、删除都是显示红色
  - **绿色**：git add 添加之后变成绿色
  - **白色**：git commit提交之后变成白色
- 项目管理--- **git add**
  - git add <file> 指令是为了用命令**git add** 将文件添加到仓库
  - **git add .** 全部进行管理操作
- 文件提交到仓库---**git commit**
  - git commit -m <message> 告诉把文件提交到仓库当中
  - **git log**---查看提交的历史记录
- 版本回滚---**git reset --hard 版本号**（提交的历史记录）
- 查看所有的提交记录---**git reflog**
- 创建分支--- **git branch**
  - git branch <分支名> 相当于两个环境没有任何影响；但是要注意要先把当前修改的分支进行提交commit之后才能正确切换分支（**git checkout**）
- 切换分支---**git checkout**
  - git checkout <分支名>
- 合并分支---**git merge**
  - git merge <分支名>
  - **git merge 合并的时候会产生冲突，此时就要进行手动修改啦（产生冲突一般就是由于修改了同一个文件的内容，所以合并的时候就会出现冲突。）**

```
将dev 中现在正在开发的功能提交到dev
git add .
git commit -m “xxxxx”
切换到主分支（master分支）
git checkout master
创建并切换到bug分支
git branch bug
git checkout bug
在bug分支上进行修复
git add .
git commit -m “xxxx”
切换回master分支，合并修复的bug分支，最后删除bug分支
git checkout master
git merge bug
git branch -d bug
查看当前有哪些分支
git branch 或git branch -a 或git branch --all

```

- 添加存储仓库地址---**git remote add** 
  - git remote add origin <仓库地址>
- 把本地仓库推送到远程仓库当中--- **git push origin master** 
  - git push origin dev 推送到分支
- 把代码从版本存储仓库下载下来---**git clone**
- 把远程存储仓库中的dev分支更新到现在的dev分支中--- **git pull origin dev**

```1、注册账户 + 创建项目 + 拷贝地址https://gitee.com/shliang/test_git.git
2、在公司本地代码推送远程

cd 项目目录
git remote add origin https://gitee.com/shliang/test_git.git
git push origin master
git push origin dev
继续写代码
git add .
git commit -m “提交记录”
git push origin dev
3、到家
下载代码
git clone https://gitee.com/shliang/test_git.git
或
创建目录
cd 目录
git init
git remote add origin https://gitee.com/shliang/test_git.git
git pull origin master
创建dev分支
git checkout dev
git pull origin dev
继续写代码
git add .
git commit -m “提交记录”
git push origin dev
```