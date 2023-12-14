#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------
# 压缩代码、去除注释
# 用于和 ChatGPT 交互，ChatGPT 不需要看这些东西
# -----------------------------------------------

import re
import sys

def compress_code(code: str) -> str:
    lines = code.split('\n')
    compressed_lines = []

    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        line = re.sub(r'\s*([=+\-*/(,)])\s*', r'\1', line)  # Remove spaces around operators and brackets
        line = re.sub(r'(#.*)$', '', line)  # 移除注释
        compressed_lines.append(line)

    compressed_code = ' '.join(compressed_lines)
    return compressed_code

def compress_file(filename: str) -> str:
    with open(filename, 'r', encoding="utf-8") as file:
        code = file.read()
    compressed_code = compress_code(code)
    return compressed_code

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    compressed = compress_file(filename)
    print(compressed)
