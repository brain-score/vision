""" Make plugin details available to readthedocs """

from pathlib import Path
from brainscore_core.plugin_management.plugin_utils import get_all_plugin_info
from brainscore_core.plugin_management.document_plugins import create_bibfile, update_readthedocs


# BIBS_DIR = Path(Path(__file__).parents[2], 'docs', 'source', 'bibtex')
PLUGINS_DOC = Path(Path(__file__).parents[2], 'docs', 'source', 'modules')
GITHUB_DIR = 'https://github.com/brain-score/vision/tree/master/brainscore_vision'


def update_docs():
    all_plugin_info = get_all_plugin_info(Path(__file__).parents[1])
    print(f"all_plugin_info: {all_plugin_info}")
    # for plugin_type in all_plugin_info:
    #     create_bibfile(all_plugin_info[plugin_type], BIBS_DIR, plugin_type) # plugin type .bib file
    # create_bibfile(all_plugin_info, BIBS_DIR) # one .bib file to rule them all
    update_readthedocs(all_plugin_info, PLUGINS_DOC, GITHUB_DIR)
            

if __name__ == '__main__':
    update_docs()
