r"""
Command-line interface to inspect optimal config(s) and metadata in the config cache.

You need to build the config signature to get the optimal config, and signature components can be
queried using the show command.

**Usage: **
::
    >>> python -m triton.tools.opt_config_cli show [--all] [--kernel ...] [--device_name ...]
"""

from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags

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
    name='device_name', default=None,
    help='Comma-separated set of device label.')


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
      '$opt_config_cli show --kernel fused_moe_kernel,awq_kernel --device_name DCU_K100_AI'
      ' [--dir ...]\n\n')
  parser_show = subparsers.add_parser(
      'show',
      description=show_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_show.set_defaults(func=show)


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
  return parser


def main():
  def smcli_main(argv):
    parser = create_parser()
    if len(argv) < 2:
      parser.error('Too few arguments.')
    args = parser.parse_args()
    args.func()

  app.run(smcli_main)


if __name__ == '__main__':
  main()
