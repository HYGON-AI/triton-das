import re
from collections import Counter

def extract_sizes_from_log(log_file):
    # 修复括号匹配的正则表达式
    pattern = r'x,x_q,x_s,group_size,N,eps,int8_min,int8_max,BLOCK,num_warps,num_stages torch\.Size\(\[(\d+), (\d+)\]\) torch\.Size\(\[\d+, \d+\]\) torch\.Size\(\[\d+, \d+\]\) (\d+)'
    sizes_counter = Counter()  # 使用 Counter 来统计频率
    
    # 使用 errors='ignore' 来忽略无法解码的字符
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                # 去除可能的 RayWorkerWrapper 前缀
                line = re.sub(r'\[.*?\] ', '', line.strip())
                
                # 严格匹配完整的日志行
                match = re.match(pattern, line)
                if match:
                    rows, cols, group_size = map(int, match.groups())
                    sizes_counter[(rows, cols, group_size)] += 1
            except Exception as e:
                print(f"Warning: Skipping problematic line due to: {e}")
                continue

    # 按频率排序
    sorted_sizes = sorted(sizes_counter.items(), key=lambda x: (-x[1], x[0]))

    # 格式化输出
    print("QUANT_TEST_CASES = [")
    print("    # rows, cols, group_size  # frequency")
    for size, freq in sorted_sizes:
        print(f"    {size},  # {freq}")
    print("]")

# 使用示例
log_file = "per_token_quant_int8_bs128_i32_o512.log"
extract_sizes_from_log(log_file)