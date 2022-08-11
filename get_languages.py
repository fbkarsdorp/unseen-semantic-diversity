
import json
import tqdm
from langdetect import detect_langs, LangDetectException
from xml.dom.minidom import parseString


def get_lines(stream):
    sent = []
    text_meta = sent_meta = None
    for line in stream:
        if line.startswith('<text'):
            attrs = parseString(line + '</text>').firstChild.attributes
            text_meta = {key: val.value for key, val in dict(attrs).items()}
        elif line.startswith('</s'):
            yield sent, sent_meta, text_meta
            sent = []
        elif line.startswith('<s'):
            attrs = parseString(line + '</s>').firstChild.attributes
            sent_meta = {key: val.value for key, val in dict(attrs).items()}
        elif line.startswith('<'):
            continue
        else:
            sent.append(line)
    if sent:
        yield sent, sent_meta, text_meta


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        with open(args.output, 'w+') as out:
            for sent, sent_meta, text_meta in tqdm.tqdm(get_lines(f)):
                tokens, lemmas = [], []
                for line in sent:
                    tok, _, lem, *_ = line.strip().split('\t')
                    tokens.append(tok)
                    lemmas.append(lem)
                try:
                    output = detect_langs(' '.join(tokens))
                    out.write(json.dumps(
                        {'lemma': lemmas, 'token': tokens,
                        'langs': [{'lang': item.lang, 'prob': item.prob} for item in output],
                        'sent_meta': sent_meta, 'text_meta': text_meta}
                    ) + "\n")
                except LangDetectException:
                    continue
                except Exception as e:
                    print(e)
