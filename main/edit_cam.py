import os
import json
import glob

def adjust_camera_params():
    # 相机参数文件目录
    cam_params_dir = "/mnt/newssd/test1/ExAvatar_RELEASE_our/avatar/data/Custom/data/cxk/cam_params"
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(cam_params_dir, "*.json"))
    
    # 按数字排序
    json_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    print(f"找到 {len(json_files)} 个相机参数文件")
    
    # 遍历每个文件
    for file_path in json_files:
        try:
            # 读取原始参数
            with open(file_path, 'r') as f:
                cam_params = json.load(f)
            
            # 调整相机位置 - 将相机调低一点
            # 这里我将t向量的y分量增加（假设y轴向上，增加y值会使相机降低）
            # 你可以根据需要调整这个值
            if 't' in cam_params and len(cam_params['t']) >= 3:
                # 将相机向下移动0.5个单位
                cam_params['t'][1] += 0.5
                
            # 写入调整后的参数
            with open(file_path, 'w') as f:
                json.dump(cam_params, f, indent=4)
            
            print(f"已调整: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print("所有相机参数调整完成！")

if __name__ == "__main__":
    adjust_camera_params()