import os
import shutil
import os.path as osp

def copy_camera_files():
    # 源文件路径
    source_file = "/mnt/newssd/test1/ExAvatar_RELEASE_our/avatar/data/Custom/data/cxk/cam/139.json"
    
    # 目标目录
    target_dir = "/mnt/newssd/test1/ExAvatar_RELEASE_our/avatar/data/Custom/data/cxk/cam"
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"错误: 源文件 {source_file} 不存在")
        return
    
    # 检查目标目录是否存在
    if not os.path.exists(target_dir):
        print(f"错误: 目标目录 {target_dir} 不存在")
        return
    
    print(f"源文件: {source_file}")
    print(f"目标目录: {target_dir}")
    
    # 获取当前目录中已有的JSON文件，找到最大的编号
    existing_files = [f for f in os.listdir(target_dir) if f.endswith('.json') and f.replace('.json', '').isdigit()]
    if existing_files:
        max_num = max(int(f.replace('.json', '')) for f in existing_files)
        print(f"当前目录中最大编号的文件: {max_num}.json")
        start_num = max_num + 1
    else:
        start_num = 0
        print("当前目录中没有JSON文件")
    
    # 设置要复制的文件数量
    num_copies = 700
    end_num = start_num + num_copies - 1
    
    print(f"将从 {start_num}.json 复制到 {end_num}.json，总共 {num_copies} 个文件")
    
    # 复制文件
    copied_count = 0
    for i in range(start_num, start_num + num_copies):
        target_file = osp.join(target_dir, f"{i}.json")
        
        try:
            shutil.copy2(source_file, target_file)
            copied_count += 1
            
            if i % 100 == 0:  # 每100个文件打印一次进度
                print(f"已复制到 {i}.json")
                
        except Exception as e:
            print(f"复制文件到 {target_file} 时出错: {e}")
    
    print(f"\n复制完成! 总共复制了 {copied_count} 个文件")
    
    # 验证复制结果
    verify_copy_result(target_dir, start_num, num_copies)

def verify_copy_result(target_dir, start_num, num_copies):
    """验证复制结果"""
    print("\n验证复制结果...")
    
    # 检查新创建的文件
    new_files = []
    for i in range(start_num, start_num + num_copies):
        file_path = osp.join(target_dir, f"{i}.json")
        if os.path.exists(file_path):
            new_files.append(file_path)
    
    print(f"验证: 成功创建了 {len(new_files)} 个新文件")
    
    if new_files:
        # 检查第一个和最后一个新文件
        first_new = osp.join(target_dir, f"{start_num}.json")
        last_new = osp.join(target_dir, f"{start_num + num_copies - 1}.json")
        
        print(f"第一个新文件: {first_new}")
        print(f"最后一个新文件: {last_new}")
        
        # 检查文件大小是否一致
        source_size = os.path.getsize("/mnt/newssd/test1/ExAvatar_RELEASE_our/avatar/data/Custom/data/cxk/cam/139.json")
        first_new_size = os.path.getsize(first_new)
        
        if source_size == first_new_size:
            print("文件大小验证: 通过")
        else:
            print(f"文件大小验证: 源文件大小 {source_size}，新文件大小 {first_new_size}")
    
    # 统计总文件数
    all_json_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
    print(f"目标目录中总共有 {len(all_json_files)} 个JSON文件")

if __name__ == "__main__":
    copy_camera_files()