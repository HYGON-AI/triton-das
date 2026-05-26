r"""
Command-line interface to inspect optimal config(s) and metadata in the config cache.

You need to build the config signature to get the optimal config, and signature components can be
queried using the show command.

**Usage: **
::
    >>> python -m triton.tools.opt_config_cli show [--all] [--kernel ...] [--device ...]
"""

import os
import json
from collections import defaultdict
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
from pathlib import Path
import shutil

import argparse

from triton import __version__ as triton_version
triton_major_version = int(triton_version.split(".")[0])
triton_minor_version = int(triton_version.split(".")[1])
triton_version_float = triton_major_version + float(triton_minor_version / 10)

from triton.backends.amd.autotuner import ConfigLoader, GraphConfigLoader, get_string_hash

def get_triton_cache_dir():
    if triton_version_float >= 3.3:
      from triton.knobs import cache as cache_knob
      return cache_knob.dir
    else:
      from triton.runtime.cache import default_cache_dir
      return os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()


# OPT CONFIG CLI flags
_OCCLI_DIR = flags.DEFINE_string(
    name='dir', default=get_triton_cache_dir(), help='Directory containing the optimal configs.')

_OCCLI_ALL = flags.DEFINE_bool(
    name='all', default=False,
    help='If set, outputs all available information in the config cache.')

_OCCLI_KERNEL = flags.DEFINE_string(
    name='kernel', default=None,
    help='kernel function name.')

_OCCLI_DEVICE_NAME = flags.DEFINE_string(
    name='device', default=None,
    help='Comma-separated set of device label.')

_OCCLI_OUTPUT_DIR = flags.DEFINE_string(
    name='output-dir', default=None, help='Output directory path.')

_OCCLI_OUTPUT = flags.DEFINE_string(
    name='output', default=None,
    help='User-provided filename of the output file.\n' \
         'Which support custom placeholder for dynamic values:\n' \
         '  %G  - When encountered, It will be replaced by the group name.\n\n' \
         'Example Usage:\n' \
         '  template = "layernorm,%G,device=K100,dtype=fp16.json"\n' \
         '  # After parsing, this might become:\n' \
         '  "layernorm,M=4096,device=K100,dtype=fp16.json"')

_OCCLI_HOIST_KEY = flags.DEFINE_string(
    name='hoist-key', default=None,
    help='Comma-separated key set (not include dtype keys) is hoisted to filename.')

_OCCLI_DELIMITER = flags.DEFINE_string(
    name='delimiter', default='-',
    help='The delimiter for the field in the file name.')

_OCCLI_HOIST_DTYPE = flags.DEFINE_bool(
    name='hoist-dtype', default=False,
    help='The dtype keyset are hoisted to filename.')

_OCCLI_KEEP_KEY = flags.DEFINE_string(
    name='keep-key', default=None,
    help='As opposed to --hoist_key')

_OCCLI_SHORT_FILENAME = flags.DEFINE_bool(
    name='short-filename', default=False,
    help='Simplify the created file name to: xxx-key=x_..._x-dtype=x_..._x. JSON')

_OCCLI_OUTPUT_FIELDS = flags.DEFINE_string(
    name='output-fields', default='config',
    help='Comma-separated set of output fields: key, config, timing, path.')

_OCCLI_GRAPH = flags.DEFINE_string(
    name='graph', default=None,
    help='Comma-separated set of triton function name.')

_OCCLI_POP_KEY = flags.DEFINE_string(
    name='pop-key', default=None,
    help='Comma-separated list of keys that will be popped from their original locations \n' \
         'and inserted into the config dictionary.')

command_required_flags = {
    'show': [],
    'export': [],
}


def _create_dict(arg_names, k):
    s = k[1:-1]
    entries = s.split(", ")
    ret = []
    for e in entries:
        if e[0] == "'" or e[0] == '"':
            ret.append(e[1:-1])
        else:
            ret.append(eval(e))
    assert len(arg_names) == len(ret)
    return dict(zip(arg_names, ret))


def to_type(t):
  if t == "'torch.float32'":
    return 'f32'
  elif t == "'torch.float16'":
    return 'f16'
  elif t == "'torch.bfloat16'":
    return 'bf16'
  elif t == "'torch.int64'":
    return 'i64'
  elif t == "'torch.int32'":
    return 'i32'
  elif t == "'torch.int8'":
    return 'i8'
  elif t.startswith("'torch.float8_"):
    return 'f8' + t[1:-1].split('_')[1]
  elif t == "'torch.bool'":
    return 'i1'
  else:
    return t


def _show_config(loader, kernel, device_name, indent=""):
  def to_config_str(dict):
    return ', '.join([f"{k}: {v}" for k, v in dict.items()])
    
  indent += "  "
  cache = loader.get_tuned_cache(kernel, device_name)

  if cache:
    config = cache.cache
    print(f"{indent}key: {config['key']}")

    print(f"{indent}configs:")
    for k, v in config['configs'].items():
      print(f"  {indent}{k}: {to_config_str(v)}")

    if _OCCLI_ALL.value:
      print(f"{indent}timings:")
      for k, v in config['timings'].items():
        print(f"  {indent}{k}: {v}")

      print(f"{indent}paths:")
      for k, v in config['paths'].items():
        print(f"  {indent}{k}: {v}")


def _show_device(loader, kernel, device_name, indent=""):
  indent += "  "
  print(f"{indent}device '{device_name}':")
  _show_config(loader, kernel, device_name, indent)


def _show_kernel(loader, kernel, indent=""):
  indent += "  "
  print(f"{indent}kernel '{kernel}':")

  if _OCCLI_DEVICE_NAME.value:
    for device_name in _OCCLI_DEVICE_NAME.value.split(","):
      _show_device(loader, kernel, device_name, indent)
  else:
    for device_name in loader.kernel_device_map[kernel]:
      _show_device(loader, kernel, device_name, indent)

  print()


def _show():
  cache_dir = os.path.join(_OCCLI_DIR.value, "configs")
  loader = ConfigLoader(cache_dir)

  print(f"The given cache dir: {cache_dir} contains the following config(s):")

  if _OCCLI_KERNEL.value:
    for name in _OCCLI_KERNEL.value.split(","):
      _show_kernel(loader, name)
  else:
    for name in loader.kernel_device_map.keys():
      _show_kernel(loader, name)


def _graph_show_config(loader, graph, device_name, indent=""):
  def to_config_str(dict):
    return ', '.join([f"{k}: {v}" for k, v in dict.items()])

  def display(i, cache):
    print(f"{indent}#{i}:")
    for k, v in cache.items():
      print(f"  {indent}{k}:")
      print(f"    {indent}key: {v['key']}")

      if _OCCLI_ALL.value:
        print(f"    {indent}config: {to_config_str(v['config'])}")
        print(f"    {indent}timings: {v['timings']}")
        print(f"    {indent}path: {v['path']}")

      print()

  indent += "  "

  caches = loader.cache[(graph, device_name)]
  for i, cache in enumerate(caches[:5]):
    display(i, cache)

  if not _OCCLI_ALL.value:
    print(f"{indent}......\n")
    print(f"{indent}Displaying the first 5 cases out of total {len(caches)} cases.")
  else:
    for i in range(5, len(caches)):
      display(i, caches[i])


def _graph_show_device(loader, graph, device_name, indent=""):
  indent += "  "
  print(f"{indent}device '{device_name}':")
  _graph_show_config(loader, graph, device_name, indent)


def _show_graph(loader, graph, indent=""):
  indent += "  "
  print(f"{indent}graph '{graph}':")

  if _OCCLI_DEVICE_NAME.value:
    for device_name in _OCCLI_DEVICE_NAME.value.split(","):
      _graph_show_device(loader, graph, device_name, indent)
  else:
    for device_name in loader.graph_device_map[graph]:
      _graph_show_device(loader, graph, device_name, indent)

  print()


def _graph_show():
  cache_dir = os.path.join(_OCCLI_DIR.value, "graph")
  loader = GraphConfigLoader(cache_dir)

  print(f"The given cache dir: {cache_dir} contains the following config(s):")

  if _OCCLI_GRAPH.value:
    for name in _OCCLI_GRAPH.value.split(","):
      _show_graph(loader, name)
  else:
    for name in loader.graph_device_map.keys():
      _show_graph(loader, name)


def show():
  """Function triggered by show command."""
  if _OCCLI_KERNEL.value and not _OCCLI_GRAPH.value:
    _show()
  elif not _OCCLI_KERNEL.value and _OCCLI_GRAPH.value:
    _graph_show()
  else:
    _show()
    _graph_show()


def _hoist_key(data, keys, hoisted):
  """
  hoists the specified keys to create groups.

  return:
  res: The configs of each groups
  group_names: group names, e.g., "N=16,K=512", which is part of filename
  """
  parsed = {}
  hoisted_ids = [keys.index(o) for o in hoisted]
  keep_ids = [i for i in range(len(keys)) if i not in hoisted_ids]
  assert keep_ids

  if not hoisted_ids:
    raise NameError(f"Not found hoisted key {hoisted}")

  for key, value in data.items():
    nums = key[1:-1].split(', ')
    parsed[tuple(nums)] = value

  grouped = defaultdict(dict)
  for k, v in parsed.items():
    gk = tuple([k[i] for i in hoisted_ids])
    kk = [eval(k[i]) if not isinstance(eval(k[i]), str) else k[i][1:-1] \
          for i in keep_ids]
    kk = str(tuple(kk)) if len(kk) > 1 else str(kk[0])
    grouped[gk][kk] = v

  res, group_names = [], []
  for k, v in grouped.items():
    res.append(v)
    kv, dtypes = [], []
    for n, _v in zip(hoisted, k):
      if isinstance(eval(_v), bool):
        kv.append(f"{n}={eval(_v)}")
      elif isinstance(eval(_v), str) and _v.startswith("'torch."):
        dtypes.append(f"{to_type(_v)}")
      else:
        kv.append(f"{n}={_v}")

    if _OCCLI_SHORT_FILENAME.value:
      kv = ['key=' + '_'.join([o.split('=')[1] for o in kv])]

    if dtypes:
      if len(set(dtypes)) == 1:
        kv.append(f"dtype={dtypes[0]}")
      else:
        kv.append(f"dtype={'_'.join(dtypes)}")
    group_names.append(_OCCLI_DELIMITER.value.join(kv))

  return res, group_names


def _get_filename(output_dir, kernel_name, device_name, group_name=''):
  """
  User-provided filename of the output file.
  Which support custom placeholder for dynamic value:
    %G  - When encountered, It will be replaced by the group name.

  Example Usage:
    template = "layernorm,%G,device=K100,dtype=fp16.json"
    # After parsing, this might become:
    "layernorm,M=4096,device=K100,dtype=fp16.json"
  """
  filename = _OCCLI_OUTPUT.value
  if not filename:
    filename = f'{kernel_name}{_OCCLI_DELIMITER.value}device={device_name}.json'

  if "%G" in filename:
    filename = filename.replace("%G", group_name)
  else:
    if group_name != '':
      # add group_name at the head of filename
      filename = Path(filename)
      filename = filename.with_stem(f'{filename.stem}{_OCCLI_DELIMITER.value}{group_name}')

  # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
  filename = Path(filename)
  filename = filename.with_stem(filename.stem[:245])

  return os.path.join(output_dir, filename)


def hoist_key(data):
  hoisted_keys = []
  paths = data['paths']
  configs = data['configs']
  keys = data['key']
  timings = data['timings']

  assert list(paths.keys()) == list(configs.keys()) == list(timings.keys())

  if _OCCLI_KEEP_KEY.value:
    keep_keys = _OCCLI_KEEP_KEY.value.split(',')
    hoisted_keys = [k for k in keys if k not in keep_keys]
  else:
    if _OCCLI_HOIST_KEY.value:
      hoisted_keys = _OCCLI_HOIST_KEY.value.split(',')

    if _OCCLI_HOIST_DTYPE.value:
      vals = list(configs.keys())[0][1:-1].split(', ')
      for k, v in zip(keys, vals):
        if isinstance(v, str) and v.startswith("'torch."):
          hoisted_keys.append(k)
    # Remove duplicate keys
    hoisted_keys = list(dict.fromkeys(hoisted_keys))

  _configs, group_names = _hoist_key(configs, keys, hoisted_keys)
  _paths, _ = _hoist_key(paths, keys, hoisted_keys)
  _timings, _ = _hoist_key(timings, keys, hoisted_keys)
  _keys = [o for o in keys if o not in hoisted_keys]
  return (_keys, _configs, _timings, _paths, group_names)


def put_json(path, data):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(data, f, indent=4)
  print(f"Generated '{path}'")


def copy_libraries(dst, src):
  for dirpath, dirnames, filenames in os.walk(src):
    for filename in filenames:
      if filename.endswith(".so"):
        src_file = os.path.join(dirpath, filename)
        rel_dir = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(dst, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_file, dst_dir)
        print(f"'{src_file}' -> '{dst_dir}/{filename}'")


def export_kernel_cache(dst_dir, data):
  src_dir = get_triton_cache_dir()
  paths = list(dict.fromkeys(data.values()))
  for p in paths:
    src = Path(f"{src_dir}/{p}")
    dst = Path(f"{dst_dir}/{p}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"'{src}' -> '{dst}'")

  # copy hip launcher and utils libraries
  copy_libraries(dst_dir, src_dir)


def get_output_fields(cache):
  fields = _OCCLI_OUTPUT_FIELDS.value.split(',')
  assert 'config' in fields, f"ERROR: --output_fields={_OCCLI_OUTPUT_FIELDS.value} not have \'config\' in it."

  if len(fields) == 1 and fields[0] == 'config':
    return cache['configs']

  res = {}
  for o in fields:
    if o == 'key':
      res['key'] = cache['key']
    elif o == 'config':
      res['config'] = cache['configs']
    elif o == 'timing':
      res['timing'] = cache['timings']
    elif o == 'path':
      res['path'] = cache['paths']
    else:
      print(f"ERROR: Unkown output field `{o}`, valid output fields: key,config,timing,path")
      raise NotImplementedError
  return res


def graph_do_pop_key(cache, keys):
  def get_key_map(keys):
    res = defaultdict(list)
    for o in keys:
      if ':' in o:
        k, v = o.split(':')
        res[k].append(v)
      else:
        res['default*'].append(o)
    return res

  # 1. pop and insert
  kmap = get_key_map(keys)
  new_cache = []
  for graph in cache:
    new_graph = {}
    for k, v in graph.items():
      if k == 'timings':
        new_graph[k] = v
      else:
        kname, s, rid = k.split('-')
        pop_keys = kmap[kname] if kname in kmap else kmap['default*']
        if pop_keys:
          bound_args = _create_dict(v['key'], s)
          for o in pop_keys:
            if o in bound_args:
              v['config'][o] = bound_args[o]
              del bound_args[o]
              v['key'].remove(o)
          new_k = f"{kname}-{str(tuple(bound_args.values()))}-{rid}"
          if new_k not in new_graph:
            new_graph[new_k] = v
        else:
          new_graph[k] = v
    new_cache.append(new_graph)

  # 2. keep the fastest graph
  graph2idx = {}
  graph2timings = defaultdict(lambda: [float("inf"), float("inf"), float("inf")])
  for i, graph in enumerate(new_cache):
    graph_hash = get_string_hash('-'.join(['-'.join(k.split('-')[:2]) for k in graph.keys()]))
    if graph['timings'] < graph2timings[graph_hash]:
      graph2timings[graph_hash] = graph['timings']
      graph2idx[graph_hash] = i

  assert len(graph2timings) == len(graph2idx)
  return [new_cache[i] for i in graph2idx.values()]


def do_pop_key(cache, pop_keys):
  if not pop_keys:
    return cache

  new_cache = {
    'key': [],
    'configs': {},
    'timings': {},
    'paths': {},
  }
  key = cache['key']
  for s in cache['configs'].keys():
    bound_args = _create_dict(key, s)
    config = cache['configs'][s].copy()
    timings = cache['timings'][s]
    path = cache['paths'][s]

    # pop and insert
    for o in pop_keys:
      if o in bound_args:
        assert not hasattr(bound_args[o], 'dtype'), f"ERROR: Can not pop a pointer {o} into config"
        config[o] = bound_args[o]
        del bound_args[o]

    new_s = str(tuple(bound_args.values()))
    if new_s not in new_cache['timings'] or timings < new_cache['timings'][new_s]:
      new_cache['configs'][new_s] = config
      new_cache['timings'][new_s] = timings
      new_cache['paths'][new_s] = path

  # update key
  new_cache['key'] = [k for k in cache['key'] if k not in pop_keys]

  return new_cache


def convert_graph_cache_to_config_cache(graph_cache):
  res = defaultdict(lambda: {
    'key': [],
    'configs': {},
    'timings': {},
    'paths': {},
  })
  for graph in graph_cache:
    for k, v in graph.items():
      if k == 'timings':
        continue
      kname, s = k.split('-')[:2]
      data = res[kname]
      if s not in data['configs'] or v['timings'] < data['timings'][s]:
        data['key'] = v['key']
        data['configs'][s] = v['config']
        data['timings'][s] = v['timings']
        data['paths'][s] = v['path']
  return res


def _graph_export():
  """Function triggered by export command."""
  cache_dir = os.path.join(_OCCLI_DIR.value, "graph")

  loader = GraphConfigLoader(cache_dir)

  if _OCCLI_OUTPUT.value:
    print(f"opt_config_cli export: warning: --output={_OCCLI_OUTPUT.value}: Flag --output is disabled in graph export, ignore it.")
    flags.FLAGS.output = None

  graph_name = _OCCLI_GRAPH.value
  device_names = []
  if _OCCLI_DEVICE_NAME.value:
    device_names.append(_OCCLI_DEVICE_NAME.value)
  else:
    for k, d in loader.cache.keys():
      if k == graph_name:
        device_names.append(d)

  output_dir = _OCCLI_OUTPUT_DIR.value if _OCCLI_OUTPUT_DIR.value else \
                os.path.join(os.getcwd(), "output")
  output_dir = os.path.join(output_dir, graph_name)
  caches = []
  for dev_name in device_names:
    cache = loader.cache[(graph_name, dev_name)]
    # export the before graph config
    put_json(_get_filename(output_dir, "before-" + graph_name, dev_name), cache)
    caches.append((dev_name, cache))

  # create total timings per graph
  for _, cache in caches:
    for graph in cache:
      timings = [0., 0., 0.]
      for _, v in graph.items():
        timings = [x + y for x, y in zip(timings, v['timings'])]
      graph['timings'] = timings

  # pop key to config
  pop_key = True if _OCCLI_POP_KEY.value else False
  if pop_key:
    keys = _OCCLI_POP_KEY.value.split(',')
    new_caches = []
    for dev_name, cache in caches:
      new_caches.append((dev_name, graph_do_pop_key(cache, keys)))
    caches = new_caches

  change_key = True if _OCCLI_HOIST_DTYPE.value or _OCCLI_HOIST_KEY.value \
                        or _OCCLI_KEEP_KEY.value else False

  new_caches = []
  for dev_name, cache in caches:
    # export the after graph config
    put_json(_get_filename(output_dir, graph_name, dev_name), cache)

    # convert graph cache to kernel config cache
    kernel_conf_cache = convert_graph_cache_to_config_cache(cache)

    # export kernel config
    for kernel_name, _cache in kernel_conf_cache.items():
      # copy kernel cache to output_dir
      # export_kernel_cache(os.path.join(output_dir, dev_name), _cache['paths'])

      if not change_key:
        data = get_output_fields(_cache)
        put_json(_get_filename(output_dir, kernel_name, dev_name), data)
      else:
        keys, configs, timings, paths, group_names = hoist_key(_cache)
        for g, c, t, p in zip(group_names, configs, timings, paths):
          data = get_output_fields({'key': keys, 'configs': c, 'timings': t, 'paths': p})
          put_json(_get_filename(output_dir, kernel_name, dev_name, g), data)


def _export():
  cache_dir = os.path.join(_OCCLI_DIR.value, "configs")
  loader = ConfigLoader(cache_dir)

  kernel_name = _OCCLI_KERNEL.value
  device_names = []
  if _OCCLI_DEVICE_NAME.value:
    device_names.append(_OCCLI_DEVICE_NAME.value)
  else:
    for k, d in loader.tuned_cache.keys():
      if k == kernel_name:
        device_names.append(d)

  caches = []
  for dev_name in device_names:
    caches.append((dev_name, loader.get_tuned_cache(kernel_name, dev_name).cache))

  output_dir = _OCCLI_OUTPUT_DIR.value if _OCCLI_OUTPUT_DIR.value else \
                os.path.join(os.getcwd(), "output")
  output_dir = os.path.join(output_dir, kernel_name)

  # pop key to config
  pop_key = True if _OCCLI_POP_KEY.value else False
  if pop_key:
    keys = _OCCLI_POP_KEY.value.split(',')
    new_caches = []
    for dev_name, cache in caches:
      new_caches.append((dev_name, do_pop_key(cache, keys)))
    caches = new_caches

  change_key = True if _OCCLI_HOIST_DTYPE.value or _OCCLI_HOIST_KEY.value \
                        or _OCCLI_KEEP_KEY.value else False

  for dev_name, _cache in caches:
    # copy kernel cache to output_dir
    # export_kernel_cache(os.path.join(output_dir, dev_name), _cache['paths'])

    if not change_key:
      data = get_output_fields(_cache)
      put_json(_get_filename(output_dir, kernel_name, dev_name), data)
    else:
      keys, configs, timings, paths, group_names = hoist_key(_cache)
      for g, c, t, p in zip(group_names, configs, timings, paths):
        data = get_output_fields({'key': keys, 'configs': c, 'timings': t, 'paths': p})
        put_json(_get_filename(output_dir, kernel_name, dev_name, g), data)


def export():
  """Function triggered by export command."""
  export_kernel = True if _OCCLI_KERNEL.value else False
  export_graph = True if _OCCLI_GRAPH.value else False

  assert (export_kernel and not export_graph) or \
         (not export_kernel and export_graph), \
         "opt_config_cli export: error: --kernel=None --graph=None: Flag --kernel or --graph must have a value other than None."

  if _OCCLI_KERNEL.value:
    _export()
  else:
    _graph_export()


def add_show_subparser(subparsers):
  """Add parser for `show`."""
  show_msg = (
      'Usage examples:\n'
      'To show all metadata in the optimal config cache:\n'
      '$opt_config_cli show [--dir /path/to/triton/config/cache/dir]\n\n'
      'To show all available information in the optimal config cache:\n'
      '$opt_config_cli show --all [--dir ...]\n\n'
      'To show all metadata of configs from specified kernel functions:\n'
      '$opt_config_cli show --kernel layernorm_kernel,awq_kernel [--dir ...]\n\n'
      'To show all metadata of configs from specified kernel functions and devices:\n'
      '$opt_config_cli show --kernel fused_moe_kernel,awq_kernel --device K100_AI'
      ' [--dir ...]\n\n')
  parser_show = subparsers.add_parser(
      'show',
      description=show_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_show.set_defaults(func=show)


def add_export_subparser(subparsers):
  """Add parser for `export`."""
  export_msg = (
      'Usage examples:\n'
      'To export the specfied kernel\'s optimal configs to JSON file:\n'
      '$opt_config_cli export --kernel _layernorm_kernel --device K100_AI'
      ' --output _layernorm_kernel-K100_AI-fp32.json'
      ' [--hoist_dtype]'
      ' [--keep_dtype key,...]'
      ' [--hoist_key key,...]'
      ' [--output_dir /path/to/save/output/file]'
      ' [--dir /path/to/triton/config/cache/dir]\n\n')
  parser_export = subparsers.add_parser(
      'export',
      description=export_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_export.set_defaults(func=export)


def create_parser():
  """Creates a parser that parse the command line arguments.

  Returns:
    A namespace parsed from command line arguments.
  """
  parser = argparse_flags.ArgumentParser(
      description='opt_config_cli: Command-line interface to inspect optimal config(s) and metadata.',
      conflict_handler='resolve')
  parser.add_argument('-v', '--version', action='version', version='0.1.0')

  subparsers = parser.add_subparsers(
      title='commands', description='valid commands', help='additional help')

  # show command
  add_show_subparser(subparsers)

  # export command
  add_export_subparser(subparsers)

  return parser


def main():
  def smcli_main(argv):
    parser = create_parser()
    if len(argv) < 2:
      parser.error('Too few arguments.')
    flags.mark_flags_as_required(command_required_flags[argv[1]])
    args = parser.parse_args()
    args.func()

  app.run(smcli_main)


if __name__ == '__main__':
  main()
