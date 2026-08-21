# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

def str_to_tuple(k):
    s = k[1:-1]
    entries = s.split(", ")
    ret = []
    for e in entries:
        if e[0] == "'" or e[0] == '"':
            ret.append(e[1:-1])
        else:
            ret.append(eval(e))
    ret_t = tuple(ret)
    return ret_t
