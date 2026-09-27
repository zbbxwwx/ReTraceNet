import run_imd_mit_full_v3_imagenet_norm  # noqa: F401; installs normalized MiT builder
import runpy

runpy.run_module("eval_imd_mit_context_locked_test", run_name="__main__")
