import argparse
import numpy as np
import os

def get_txt_file_paths(folder_path):
    txt_file_paths = []

    # 遍历主文件夹
    for root, dirs, files in os.walk(folder_path):
        # 遍历子文件夹
        for file in files:
            # 检查文件是否以'.txt'为扩展名
            if file.endswith('.txt'):
                # 构造txt文件的完整路径
                txt_file_path = os.path.join(root, file)
                # 将路径添加到列表中
                txt_file_paths.append(txt_file_path)

    return txt_file_paths

def has_min_lines(file_path, min_lines=3500):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取文件的行数
            lines = file.readlines()
            return len(lines) >= min_lines
    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在。")
        return False
    except Exception as e:
        print(f"发生错误：{e}")
        return False


if __name__ == '__main__':
    try:

        dataset = 'MUTAG' 
        epochs = 350

        result_acc = []
        result_std = []
    
        # 替换'your_folder_path'为你的文件夹路径
        folder_path = './results/'+dataset
        txt_files = get_txt_file_paths(folder_path)
     
        file_num = 0
        file_list = []
        
        # txt_files = ['']
        # 打印所有txt文件的路径
        folds = 10
        for txt_file in txt_files:
            validation_loss = np.zeros((epochs,folds))
            test_accuracy = np.zeros((epochs, folds))
            test_acc = np.zeros(folds)
            
          
            if has_min_lines(txt_file, min_lines=350*folds):
                file_list.append(file_num)
                with open(txt_file, 'r') as filehandle:
                    filecontents = filehandle.readlines()
                    index = 0
                    col = 0
                    for line in filecontents:
                        ss = line.split()
                        t_acc = ss[1]
                        v_loss = ss[0]
                        validation_loss[index][col] = float(v_loss)
                        test_accuracy[index][col] = float(t_acc)
                        index += 1
                        if index == epochs:
                            index = 0
                            col += 1
                            if col == folds:
                                break

                min_ind = np.argmin(validation_loss, axis=0)
                for i in range(folds):
                    ind = min_ind[i]
                    test_acc[i] = test_accuracy[ind][i]
                ave_acc = np.mean(test_acc)
                std_acc = np.std(test_acc)
                result_acc.append(ave_acc)
                result_std.append(std_acc)
            file_num = file_num + 1


        sorted_list = sorted(enumerate(result_acc), key=lambda x: x[1])
        top_10_values = sorted_list[-10:][::-1]
        for position, value in top_10_values:
            file_name = txt_files[file_list[position]]
            best_std = result_std[position]
            print('test accuracy / mean(std): {0:.5f}({1:.2f})'.format(value, best_std*100))
            print('file_name: {}'.format(file_name))
    except IOError as e:
        print(e)
