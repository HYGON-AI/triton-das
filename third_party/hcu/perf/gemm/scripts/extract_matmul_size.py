import re
from collections import Counter

def extract_sizes_from_log(log_file):
    # 匹配完整的日志行格式
    pattern = r'A,B,As,Bs,block_size0,block_size1 torch\.Size\(\[(\d+), (\d+)\]\) torch\.Size\(\[(\d+), \d+\]\) torch\.Size\(\[\d+, \d+\]\) torch\.Size\(\[\d+, \d+\]\) (\d+) (\d+)'
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
                    M, K, N, block_size0, block_size1 = map(int, match.groups())
                    # 构建测试用例格式：(M, K, N, (block_size0, block_size1))
                    # 使用元组而不是列表，因为元组是可哈希的
                    test_case = (M, K, N, (block_size0, block_size1))
                    sizes_counter[test_case] += 1
            except Exception as e:
                print(f"Warning: Skipping problematic line due to: {e}")
                continue

    # 按频率排序
    sorted_sizes = sorted(sizes_counter.items(), key=lambda x: (-x[1], x[0]))

    # 格式化输出
    print("MATMUL_TEST_CASES = [")
    print("    # M, K, N, block_size  # frequency")
    for (M, K, N, (block_size0, block_size1)), freq in sorted_sizes:
        # 输出时将元组转换回列表格式
        print(f"    ({M}, {K}, {N}, [{block_size0}, {block_size1}]),  # {freq}")
    print("]")

# 使用示例
log_file = "w8a8_block_int8_matmul_bs128_i32_o512.log"
extract_sizes_from_log(log_file) 