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
import torch
from collections import defaultdict
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
from pathlib import Path

import argparse
from triton.utils.hcutuner import get_config_cache_dir, ConfigLoader


# OPT CONFIG CLI flags
_OCCLI_DIR = flags.DEFINE_string(
    name='dir', default=get_config_cache_dir(), help='Directory containing the optimal configs.')

_OCCLI_ALL = flags.DEFINE_bool(
    name='all', default=False,
    help='If set, outputs all available information in the config cache.')

_OCCLI_KERNEL = flags.DEFINE_string(
    name='kernel', default=None,
    help='Comma-separated set of triton function name.')

_OCCLI_DEVICE_NAME = flags.DEFINE_string(
    name='device', default=None,
    help='Comma-separated set of device label.')

_OCCLI_OUTPUT_DIR = flags.DEFINE_string(
    name='output_dir', default=None, help='Output directory path.')

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
    name='hoist_key', default=None,
    help='Comma-separated key set (not include dtype keys) is hoisted to filename.')

_OCCLI_DELIMITER = flags.DEFINE_string(
    name='delimiter', default='-',
    help='The delimiter for the field in the file name.')

_OCCLI_HOIST_DTYPE = flags.DEFINE_bool(
    name='hoist_dtype', default=False,
    help='The dtype keyset are hoisted to filename.')

_OCCLI_KEEP_KEY = flags.DEFINE_string(
    name='keep_key', default=None,
    help='As opposed to --hoist_key')

command_required_flags = {
    'show': [],
    'export': ['kernel', 'device', 'output'],
}


def to_type(t):
  if t == "'torch.float32'":
    return 'fp32'
  elif t == "'torch.float16'":
    return 'fp16'
  elif t == "'torch.bfloat16'":
    return 'bf16'
  elif t == "'torch.int32'":
    return 'i32'
  elif t == "'torch.int8'":
    return 'i8'
  elif t == "'torch.'":
    return 'bf16'
  elif t.startswith("'torch.float8_'"):
    return 'fp8'
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

    if _OCCLI_ALL.value:
      print(f"{indent}configs:")
      for k, v in config['configs'].items():
        print(f"  {indent}{k}: {to_config_str(v)}")

      print(f"{indent}timings:")
      for k, v in config['timings'].items():
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


def show():
  """Function triggered by show command."""
  loader = ConfigLoader(_OCCLI_DIR.value)

  print(f"The given cache dir: {_OCCLI_DIR.value} contains the following config(s):")

  if _OCCLI_KERNEL.value:
    for name in _OCCLI_KERNEL.value.split(","):
      _show_kernel(loader, name)
  else:
    for name in loader.kernel_device_map.keys():
      _show_kernel(loader, name)


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
    if dtypes:
      if len(set(dtypes)) == 1:
        kv.append(f"dtype={dtypes[0]}")
      else:
        kv.append(f"dtype={'_'.join(dtypes)}")
    group_names.append(_OCCLI_DELIMITER.value.join(kv))

  return res, group_names


def _get_filename(output_dir, group_name=''):
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
  if "%G" in filename:
    filename = filename.replace("%G", group_name)
  else:
    if group_name != '':
      # add group_name at the head of filename
      filename = Path(filename)
      filename = filename.with_stem(f'{filename.stem}{_OCCLI_DELIMITER.value}{group_name}')

  return os.path.join(output_dir, filename)


def export():
  """Function triggered by export command."""
  loader = ConfigLoader(_OCCLI_DIR.value)
  _cache = loader.get_tuned_cache(_OCCLI_KERNEL.value, _OCCLI_DEVICE_NAME.value).cache

  output_dir = _OCCLI_OUTPUT_DIR.value if _OCCLI_OUTPUT_DIR.value else os.getcwd()
  os.makedirs(output_dir, exist_ok=True)

  change_key = True if _OCCLI_HOIST_DTYPE.value or _OCCLI_HOIST_KEY.value \
                        or _OCCLI_KEEP_KEY.value else False

  if not change_key:
    with open(_get_filename(output_dir), "w") as f:
      json.dump(_cache['configs'], f, indent=4)
  else:
    if _OCCLI_KEEP_KEY.value:
      keep_keys = _OCCLI_KEEP_KEY.value.split(',')
      hoisted_keys = [k for k in _cache['key'] if k not in keep_keys]
    else:
      hoisted_keys = set()
      if _OCCLI_HOIST_KEY.value:
        for k in _OCCLI_HOIST_KEY.value.split(','):
          hoisted_keys.add(k)

      if _OCCLI_HOIST_DTYPE.value:
        keys = _cache['key']
        vals = list(_cache['configs'].keys())[0][1:-1].split(', ')
        for k, v in zip(keys, vals):
          if isinstance(v, str) and v.startswith("'torch."):
            hoisted_keys.add(k)
      hoisted_keys = list(hoisted_keys)

    configs, group_names = _hoist_key(_cache['configs'], _cache['key'], hoisted_keys)
    for g, config in zip(group_names, configs):
      with open(_get_filename(output_dir, g), "w") as f:
        json.dump(config, f, indent=4)


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
      '$opt_config_cli show --kernel fused_moe_kernel,awq_kernel --device DCU_K100_AI'
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
      '$opt_config_cli export --kernel _layernorm_kernel --device DCU_K100_AI'
      ' --output _layernorm_kernel-DCU_K100_AI-fp32.json'
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
