"""
Creates document id files where each row id represents
where a document starts in each corpus. This is necessary
for running mmax converter.
"""

with open('doc-ids.csv') as f:
    with open('doc-starts-at.csv', 'w') as w:
        docid = "-1"
        for i, did in enumerate(f):
            if did.split('\t')[0].strip() != docid.strip():
                docid = did.split()[0].strip()
                w.write(str(i) + '\n')
