import re
from collections import defaultdict


class Sentence():

    def __init__(self, source_sentence, target_sentence, classes, removed_words, alignments):
        self.source_sentence = tokenize(source_sentence)
        self.target_sentence = tokenize(target_sentence)
        self.source_word2index = get_word_indices(source_sentence)
        self.target_word2index = get_word_indices(source_sentence)
        self.classes = classes
        self.removed_words = removed_words
        self.alignments = alignments
        self.source2target_alignments, self.target2source_alignments = aggregate_alignments(alignments)

    @property
    def removed_words_target_indices(self):
        indices = []
        for i, word in enumerate(self.target_sentence):
            if word == "REPLACE":
                indices.append(i)
        return indices

    @property
    def removed_words_source_indices(self):
        return [self.target2source_alignments[w] for w in self.removed_words_target_indices]

    @property
    def source_words_removed(self):
        return [[self.source_sentence[source_index] for source_index in self.target2source_alignments[w]]
                for w in self.removed_words_target_indices]

    def source2targets(self, word):
        source_indices = self.source_word2index[word]
        target_words = []
        for source_index in source_indices:
            target_words.append(self.source2target_alignments[source_index])
        return target_words

    def target2sources(self, word):
        target_indices = self.target_word2index[word]
        source_words = []
        for target_index in target_indices:
            source_words.append(self.target2source_alignments[target_index])
        return source_words

    def removed_words_target_contexts(self, left_context, right_context):
        contexts = []
        for target_index in self.removed_words_target_indices:
            context = []
            # Add sentence start symbols when out-of-bounds
            for _ in range(max(0, left_context - target_index)):
                context.append("<S>")

            for i in range(-left_context, right_context + 1):
                if target_index + i >= 0 and target_index + i < len(self.target_sentence):
                    context.append(self.target_sentence[target_index + i])

            # Add sentence end symbols when out-of-bounds
            for _ in range(max(0, 1 + right_context + target_index - len(self.target_sentence))):
                context.append("<E>")

            contexts.append(context)

        return contexts

    def removed_words_source_contexts(self, left_context, right_context):
        contexts = []
        for source_indices in self.removed_words_source_indices:
            # the source might be more than one word, so we have to deal with this special case
            end_index = len(source_indices) - 1
            context = []
            # Add sentence start symbols when out-of-bounds
            for _ in range(max(0, left_context - source_indices[0])):
                context.append("<S>")

            for i in range(-left_context, right_context + end_index + 1):
                if source_indices[0] + i >= 0 or source_indices[0] + end_index + i < len(self.source_sentence):
                    context.append(self.source_sentence[source_indices[0] + i])

            # Add sentence end symbols when out-of-bounds
            for _ in range(max(0, right_context + source_indices[0] + end_index - len(self.target_sentence))):
                context.append("<E>")

            contexts.append(context)

        return contexts

    def __str__(self):
        line = " ".join(self.classes) + "\n" + " ".join(self.removed_words)
        line += "\n" + " ".join(self.source_sentence) + "\n" + " ".join(self.target_sentence)
        line += "\n" + " ".join(self.alignments)
        return line

    def __repr__(self):
        return self.__str__()


def tokenize(sentence):
    tokenized = []
    for word in sentence.split():
        if word.startswith("REPLACE_"):
            tokenized.append("REPLACE")
        else:
            tokenized.append(word.lower().strip())
    return tokenized


def get_word_indices(sentence):
    word_indices = defaultdict(set)
    for index, word in enumerate(sentence):
        word_indices[word].add(index)
    return word_indices


def aggregate_alignments(align_tokens):
    """
    Parse the alignment file.

    Return:
      - s2t: a dict mapping source position to target position
      - t2s: a dict mapping target position to source position
    """
    s2t = {}
    t2s = {}
    # process alignments
    for align_token in align_tokens:
        if align_token == '':
            continue
        (src_pos, tgt_pos) = re.split('\-', align_token)
        src_pos = int(src_pos)
        tgt_pos = int(tgt_pos)
        if src_pos not in s2t:
            s2t[src_pos] = []
        s2t[src_pos].append(tgt_pos)

        if tgt_pos not in t2s:
            t2s[tgt_pos] = []
        t2s[tgt_pos].append(src_pos)

    return s2t, t2s
