import json
from pkg_resources import resource_string


def get_ocrd_tool():
    return json.loads(
        resource_string(__name__, 'ocrd-tool.json').decode('utf8'))
