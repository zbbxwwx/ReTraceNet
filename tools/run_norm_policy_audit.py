import run_imd_mit_full_v3_imagenet_norm  # noqa: F401; patches normalized MiT builder
import runpy

runpy.run_module("eval_imd_mit_context_policies", run_name="__main__")
