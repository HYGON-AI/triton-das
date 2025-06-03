import re
import json
import os
import argparse

def parse_log_file(log_path):
    configs = {}
    current_size = None
    found_kernel = False
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # Look for kernel marker
        if '_w8a8_block_int8_matmul:' in line:
            found_kernel = True
            continue
            
        # Look for size configurations after kernel marker
        if found_kernel and 'Configs:' in line and '(' in line:
            try:
                # Extract M, K, N from the line
                size_match = re.search(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),', line)
                if size_match:
                    M, K, N, block_n, block_k = map(int, size_match.groups())
                    current_size = (M, K, N, block_n, block_k)
                    found_kernel = False  # Reset for next kernel section
            except:
                continue
                
        # Look for best config selections
        if 'best config selected:' in line:
            if current_size is None:
                continue
                
            M, K, N, block_n, block_k = current_size
            
            # Extract configuration parameters
            config_str = line.split('best config selected:')[1].strip()
            print(f"Config string: {config_str}")  # Debug print
            # Strip whitespace from keys when creating dictionary
            params = {k.strip(): v.strip() for k, v in (item.split(':') for item in config_str.split(',') if ':' in item)}
            print(f"Parsed params: {params}")  # Debug print
            
            try:
                # Clean up parameter values - no default values
                config = {
                    "BLOCK_SIZE_M": int(params['BLOCK_SIZE_M']),
                    "BLOCK_SIZE_N": int(params['BLOCK_SIZE_N']),
                    "BLOCK_SIZE_K": int(params['BLOCK_SIZE_K']),
                    "GROUP_SIZE_M": int(params['GROUP_SIZE_M']),
                    "num_warps": int(params['num_warps']),
                    "num_stages": int(params['num_stages']),
                    "COMBINE_SCALE_LOAD": 1 if params.get('COMBINE_SCALE_LOAD', '').lower() == 'true' else 0
                }
            except KeyError as e:
                print(f"Missing key in params: {e}")  # Debug print
                continue
            
            # Use K-N and block_size from current_size as key for grouping
            key = (K, N, block_n, block_k)  # Use block sizes from current_size for filename
            if key not in configs:
                configs[key] = {}
            configs[key][M] = config  # Store only the config from best config
            
            current_size = None
    
    return configs

def save_configs_to_json(configs, device_name, do_not_update_existing_configs=False):
    os.makedirs('configs', exist_ok=True)
    
    for (K, N, block_n, block_k), m_configs in configs.items():
        filename = f"N={N},K={K},device_name={device_name},dtype=int8_w8a8,block_shape=[{block_n}, {block_k}].json"
        filepath = os.path.join('configs', filename)
        
        existing_configs = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing_configs = json.load(f)
                    
                if do_not_update_existing_configs:
                    # 只添加不存在的 M 值对应的配置，保留已存在的配置不更新
                    for M, config in m_configs.items():
                        if str(M) not in existing_configs:
                            existing_configs[str(M)] = config
                    m_configs_to_save = existing_configs
                else:
                    # 默认行为：合并所有配置，新配置会覆盖同 M 值的旧配置
                    m_configs_to_save = existing_configs
                    m_configs_to_save.update({str(M): config for M, config in m_configs.items()})
            except json.JSONDecodeError:
                # 如果文件无法解析，就使用新的配置
                m_configs_to_save = {str(M): config for M, config in m_configs.items()}
        else:
            # 不存在配置文件时，使用新的配置
            m_configs_to_save = {str(M): config for M, config in m_configs.items()}
        
        # 对配置按照 M 值进行排序
        sorted_configs = dict(sorted(m_configs_to_save.items(), key=lambda x: int(x[0])))
                
        with open(filepath, 'w') as f:
            json.dump(sorted_configs, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Parse performance log file and generate config files')
    parser.add_argument('log_path', help='Path to the log file to parse')
    parser.add_argument('--do-not-update-existing-configs', action='store_true', 
                      help='Only add new M values while preserving existing configurations (skip updating existing M values)')
    args = parser.parse_args()
    
    device_name = "K100_AI"  # Device name set to K100-AI
    configs = parse_log_file(args.log_path)
    save_configs_to_json(configs, device_name, args.do_not_update_existing_configs)

if __name__ == "__main__":
    main() 
