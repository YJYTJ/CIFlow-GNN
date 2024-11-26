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
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--dataset", help="dataset name")
        # args = parser.parse_args()
        dataset = 'MUTAG'
        epochs = 350
        # layers = [3]
        # batches = [32,64,128]
        # learning_rates = [0.01,0.001,0.0001]
        result_acc = []
        result_std = []
    
        # 替换'your_folder_path'为你的文件夹路径
        folder_path = 'results/'+dataset
        txt_files = get_txt_file_paths(folder_path)
        # txt_files_new = []
        # for i in range(len(txt_files)):
        #     if 'step3' in txt_files[i]:
        #         txt_files_new.append(txt_files[i])
        # txt_files = txt_files_new
        file_num = 0
        file_list = []
        
        # txt_files = ['']
        # 打印所有txt文件的路径
        for txt_file in txt_files:
            validation_loss = np.zeros((epochs, 10))
            test_accuracy = np.zeros((epochs, 10))
            test_acc = np.zeros(10)
            
            # with open(results_folder+'/{}_acc_results.txt'.format(args.dataset), 'r') as filehandle:
            if has_min_lines(txt_file, min_lines=3500):
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
                            if col == 10:
                                break

                min_ind = np.argmin(validation_loss, axis=0)
                for i in range(10):
                    ind = min_ind[i]
                    test_acc[i] = test_accuracy[ind][i]
                ave_acc = np.mean(test_acc)
                std_acc = np.std(test_acc)
                # print(ave_acc)
                result_acc.append(ave_acc)
                result_std.append(std_acc)
            file_num = file_num + 1
        # best_acc = max(result_acc)
        # idx = np.argmax(result_acc)
        # result_acc = np.array(result_acc)
        # max_positions = np.where(result_acc > 0)[0]
        # for idx in max_positions:
        #     file_name = txt_files[file_list[idx]]
        #     acc = result_acc[idx]
        #     print('test accuracy / mean(std): {0:.5f}'.format(acc))
        #     print('file_name: {}'.format(file_name))

        sorted_list = sorted(enumerate(result_acc), key=lambda x: x[1])
        top_10_values = sorted_list[-10:][::-1]
        for position, value in top_10_values:
            file_name = txt_files[file_list[position]]
            best_std = result_std[position]
            print('test accuracy / mean(std): {0:.5f}({1:.2f})'.format(value, best_std*100))
            print('file_name: {}'.format(file_name))
    except IOError as e:
        print(e)

# /3layer/MUTAG/learning_rate0.01/batchsize32/max_step1/1_size_graph_filter[4,4,6]_hidden_dims[16,32,32,64]_ntklayer1_acc_results.txt'