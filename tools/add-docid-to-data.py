with open('data.csv') as data, open('doc-starts-at.csv') as doc, open('data-with-doc.csv', "w") as w:
    doc_line = int(next(doc))
    for i, line in enumerate(data):
        if i == doc_line:
            new_doc = 1
            print("New doc at {}".format(doc_line))
            try:
                doc_line = int(next(doc))
            except StopIteration:
                print("Last document at {}".format(doc_line))
        else:
            new_doc = 0
        w.write(str(new_doc) + '\t' + line)
