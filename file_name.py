import os

folder_path = r"D:\project_D\audio"

files = os.listdir(folder_path)

files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

for file in files:
    print(file)
