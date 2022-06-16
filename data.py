
import os
import lxml.etree as etree

if __name__ == '__main__':
    root = 'data/shakespeares-works_TEIsimple_FolgerShakespeare'
    target = 'data/shakespeares-works_TEIsimple_FolgerShakespeare.txt'
    files = os.listdir(root)

    for f in files:
        path = os.path.join(root, f)

        tree = etree.parse(path).getroot()
        nodes = tree.xpath(
            "(//tei:l/tei:w|//tei:l/tei:pc|//tei:p/tei:w|//tei:p/tei:pc)", 
            namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
        text = []
        for node in nodes:
            if node.text is None:
                print("!")
                continue
            text.append({'lemma': node.attrib.get('lemma'), 'form': node.text.strip()})

        output_path = '.'.join(f.split('.')[:-1]) + '.txt'
        output_path = os.path.join(target, output_path)
        with open(output_path, 'w') as out:
            for node in text:
                out.write('\t'.join([node['form'], node['lemma'] or ' ']) + '\n')

