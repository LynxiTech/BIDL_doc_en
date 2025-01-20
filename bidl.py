import os


def replace_in_files(folder_path):
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.rst'):
                file_path = os.path.join(root, file)
                # 读取文件内容
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    if 'LYNBIDL' in line:
                        print(line.strip())  # 打印包含被替换部分的行
                    new_lines.append(line.replace('LYNBIDL', 'BIDL'))
                # 将修改后的内容覆盖原文件
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)


if __name__ == "__main__":
    # 请将此路径修改为你需要处理的文件夹的实际路径
    folder_path = 'source'
    replace_in_files(folder_path)
