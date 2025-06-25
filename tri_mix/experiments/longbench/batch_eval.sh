#!/bin/bash

# 检查是否提供了路径参数
if [ -z "$1" ]; then
    echo "用法: $0 <目录路径>"
    exit 1
fi

# 获取输入的目录路径
DIR_PATH="$1"

# 确保提供的路径是一个目录
if [ ! -d "$DIR_PATH" ]; then
    echo "错误: $DIR_PATH 不是一个目录"
    exit 1
fi

# 遍历子文件夹并输出绝对路径
for dir in "$DIR_PATH"/*/; do
    if [ -d "$dir" ]; then
        python eval.py --model `realpath "$dir"`
    fi
done