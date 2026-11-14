"""The three places that name the version and the plugin class agree. No torch, no GPU.

    python3 -m unittest tests.test_package -v
"""

import ast
import os
import re
import tomllib
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


class Package(unittest.TestCase):
    def setUp(self):
        self.project = tomllib.loads(read("pyproject.toml"))["project"]
        self.run_sh = read("mod", "run.sh")

    def test_one_version_in_three_places(self):
        in_package = re.search(r'^__version__ = "([^"]+)"', read("mod", "vllm_overlay_switch", "__init__.py"), re.M).group(1)
        in_installer = re.search(r"^VERSION=(\S+)", self.run_sh, re.M).group(1)
        self.assertEqual({self.project["version"], in_package, in_installer}, {in_package})

    def test_the_plugin_is_registered_the_same_way_by_both_installers(self):
        target = self.project["entry-points"]["vllm.endpoint_plugins"]["overlay_switch"]
        self.assertIn(f"overlay_switch = {target}", self.run_sh)
        module, name = target.split(":")
        self.assertEqual(module, "vllm_overlay_switch.api")
        tree = ast.parse(read("mod", "vllm_overlay_switch", "api.py"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn(name, classes)
        # the name vLLM enables plugins by is the entry point's name
        assigned = {t.id: n.value.value for n in classes[name].body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
        self.assertEqual(assigned.get("name"), "overlay_switch")

    def test_the_worker_class_named_in_the_docs_exists(self):
        tree = ast.parse(read("mod", "vllm_overlay_switch", "worker.py"))
        self.assertIn("OverlayWorkerExtension", {n.name for n in tree.body if isinstance(n, ast.ClassDef)})
        for text in (self.run_sh, read("README.md"), read("examples", "recipe-additions.yaml")):
            self.assertIn("vllm_overlay_switch.worker.OverlayWorkerExtension", text)

    def test_the_worker_adds_only_its_three_names(self):
        # the engine refuses an extension whose attributes collide with its worker's
        tree = ast.parse(read("mod", "vllm_overlay_switch", "worker.py"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OverlayWorkerExtension")
        names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(names, ["overlay_status", "overlay_prepare", "overlay_switch"])


if __name__ == "__main__":
    unittest.main()
