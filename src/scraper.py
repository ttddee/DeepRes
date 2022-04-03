from jmd_imagescraper.core import * # dont't worry, it's designed to work with import *
from pathlib import Path

root = Path().cwd().parent

duckduckgo_search(root, "download/LeafMold", "tomato leaf mold", max_results=500)