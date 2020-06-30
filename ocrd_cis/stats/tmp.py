from __future__ import absolute_import

from ocrd.model.ocrd_page_generateds import parse

if __name__ == "__main__":
    doc = parse("tmp.xml", True)
    print(doc.get_Page().get_TextRegion()[0].get_TextLine()[0].get_TextEquiv()[0].Unicode)
    print(doc.get_Page().get_TextRegion()[0].get_TextLine()[0].get_TextEquiv()[1].Unicode)
