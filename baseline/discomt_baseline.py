#!/usr/bin/python
# -*- coding:UTF-8 -*-
import sys
import os
import re
import math
import optparse
import kenlm
from collections import defaultdict
from gzip import GzipFile

'''
takes a classification file (optionally gzipped) as input
and writes a file with two columns (replacement+target)

Input format:
  - classes (ignored)
  - true replacements (ignored)
  - source text (currently ignored)
  - target text
  - alignments (currently ignored)

Output format (for fmt=replace):
  - predicted classes
  - predicted replacements
  - original source text
  - original target text
  - alignments
'''

oparse = optparse.OptionParser(usage='%prog [options] input_file')
oparse.add_option('--lm', dest='lm',
                  help='language model file',
                  default='corpus.5.fr.trie.kenlm')
oparse.add_option('--fmt', dest='fmt',
                  choices=['replace', 'predicted', 'both', 'compare', 'scores'],
                  help='format (replace, predicted, both, compare, scores)',
                  default='replace')
oparse.add_option('--none-penalty', dest='none_penalty',
                  type='float', default=0.0,
                  help='penalty for empty filler')

replace_re = re.compile('REPLACE_[0-9]+')

all_fillers = [
    ['il'], ['elle'],
    ['ils'], ['elles'],
    ["c'"], ["ce"], ["ça"], ['cela'], ["on"]]

non_fillers = [[w] for w in
               '''
               le l' se s' y en qui que qu' tout
               faire ont fait est parler comprendre chose choses
               ne pas dessus dedans
               '''.strip().split()]

def map_class(x):
    if [x] in non_fillers:
        return 'OTHER'
    elif x == 'NONE':
        return 'OTHER'
    elif x == "c'":
        return 'ce'
    else:
        return x

NONE_PENALTY = 0

def gen_items(contexts, prev_contexts):
    '''
    extends the items from *prev_contexts* with
    fillers and the additional bits of context from
    *contexts*

    returns a list of (text, score, fillers) tuples,
    and expects prev_contexts to have the same shape.
    '''
    if len(contexts) == 1:
        return [(x+contexts[0], y, z)
                for (x,y,z) in prev_contexts]
    else:
        #print >>sys.stderr, "gen_items %s %s"%(contexts, prev_contexts)
        context = contexts[0]
        next_contexts = []
        for filler in all_fillers:
            next_contexts += [(x+context+filler, y, z+filler)
                              for (x,y,z) in prev_contexts]
        for filler in non_fillers:
            next_contexts += [(x+context+filler, y, z+filler)
                              for (x,y,z) in prev_contexts]
        next_contexts += [(x+context, y+NONE_PENALTY, z+['NONE'])
                            for (x,y,z) in prev_contexts]
        if len(next_contexts) > 5000:
            print >>sys.stderr, "Too many alternatives, pruning some..."
            next_contexts = next_contexts[:200]
            next_contexts.sort(key=score_item, reverse=True)
        return gen_items(contexts[1:], next_contexts)

def score_item(x):
    model_score = model.score(' '.join(x[0]))
    return model_score + x[1]

def main(argv=None):
    global model, NONE_PENALTY
    opts, args = oparse.parse_args(argv)
    if not args:
        oparse.print_help()
        sys.exit(1)
    NONE_PENALTY = opts.none_penalty
    discomt_file = args[0]
    print >>sys.stderr, "Loading language model..."
    model = kenlm.LanguageModel(opts.lm)
    mode = opts.fmt
    print >>sys.stderr, "Processing stuff..."
    if discomt_file.endswith('.gz'):
        f_input = GzipFile(discomt_file)
    else:
        f_input = file(discomt_file)
    for i, l in enumerate(f_input):
        if l[0] == '\t':
            if mode == 'replace':
                print l,
                continue
            elif mode != 'scores':
                continue
        classes_str, target, text_src, text, text_align = l.rstrip().split('\t')
        if mode == 'scores':
            print '%d\tTEXT\t%s\t%s\t%s' % (i, text_src, text, text_align)
            if l[0] == '\t':
                continue
        text = replace_re.sub('REPLACE', text)
        targets = [x.strip() for x in target.split(' ')]
        classes = [x.strip() for x in classes_str.split(' ')]
        contexts = [x.strip().split() for x in text.split('REPLACE')]
        #print "TARGETs:", target
        #print "CONTEXTs: ", contexts
        if len(contexts) > 5:
            print >>sys.stderr, "#contexts:", len(contexts)
        items = gen_items(contexts, [([], 0.0, [])])
        items.sort(key = score_item, reverse=True)
        pred_fillers = items[0][2]
        pred_classes = [map_class(x) for x in pred_fillers]
        if mode == 'scores':
            #TODO compute individual scores for each slot
            # and convert the scores to probabilities
            scored_items = []
            for item in items:
                words, penalty, fillers = item
                scored_items.append((words, score_item(item), fillers))
            best_penalty = max([x[1] for x in items])
            dists = [defaultdict(float) for k in items[0][2]]
            for words, penalty, fillers in scored_items:
                exp_pty = math.exp(penalty - best_penalty)
                for j, w in enumerate(fillers):
                    dists[j][w] += exp_pty
            for j in xrange(len(items[0][2])):
                sum_all = sum(dists[j].values())
                if sum_all == 0:
                    sum_all = 1.0
                items = [(k, v/sum_all) for k,v in dists[j].iteritems()]
                items.sort(key=lambda x: -x[1])
                print "%s\tITEM %d\t%s"%(
                    i, j, ' '.join([
                        '%s %.4f'%(x[0], x[1])
                        for x in items if x[1] > 0.001]))
        elif mode == 'both':
            print "%s\t%s"%(target, ' '.join(pred_fillers))
        elif mode == 'predicted':
            print "%s\t%s"%(
                ' '.join(pred_classes),
                ' '.join(pred_fillers))
        elif mode == 'replace':
            print "%s\t%s\t%s\t%s\t%s"%(
                ' '.join(pred_classes),
                ' '.join(pred_fillers),
                text_src, text, text_align)
        elif mode == 'compare':
            assert len(classes) == len(pred_classes), (classes, pred_classes)
            for gold, syst in zip(classes, pred_classes):
                print "%s\t%s"%(gold, syst)

if __name__ == '__main__':
    main()
