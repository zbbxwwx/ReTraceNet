import run_imd_mit_full_v3_imagenet_norm  # noqa: F401; normalized MiT + 1024/768 evaluator patch
import runpy

runpy.run_module("evaluate_mit_full_external", run_name="__main__")
