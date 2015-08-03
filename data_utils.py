import re
from collections import defaultdict
from textblob.en import parser


class Sentence():

    """
    A sentence representation. This contains the following features:
    - Source language sentence.
    - Target language sentence.
    - Classes.
    - Word alignments.
    - Pointer to preceding sentence, if there is one.
    - What words are removed.
    - Preceding POS tags.
    """

    def __init__(self, source_sentence, target_sentence,
                 alignments=None, classes=None, removed_words=None, prev_sentence=None):
        self.source_sentence = tokenize(source_sentence)
        self.target_sentence = tokenize(target_sentence)
        self.source_word2index = get_word_indices(source_sentence)
        self.target_word2index = get_word_indices(source_sentence)
        self.classes = classes
        self.removed_words = removed_words
        self.alignments = alignments
        self.source2target_alignments, self.target2source_alignments = aggregate_alignments(alignments)
        self.prev_sentence = prev_sentence

    @property
    def source_pos_tags(self):
        if not hasattr(self, "_source_pos_tags"):
            tags = parser.find_tags(self.source_sentence)
            self._source_pos_tags = [word_tag[1] for word_tag in tags]
        return self._source_pos_tags

    def get_previous_indices_with_tag(self, word_index, n_tags, tags=("NN",), doc_id=0):
        indices = []
        for i, word_tag in enumerate(self.source_pos_tags[word_index - 1::-1]):
            index = word_index - i - 1
            if word_tag in tags:
                indices.append((index, doc_id))

        if len(indices) < n_tags and self.prev_sentence is not None:
            indices += self.prev_sentence.get_previous_indices_with_tag(len(self.prev_sentence.source_sentence) - 1,
                                                                        n_tags - len(indices),
                                                                        tags=tags,
                                                                        doc_id=doc_id + 1)
        if len(indices) < n_tags:
            indices += [("N/A", "N/A")] * (n_tags - len(indices))

        return indices[:n_tags]

    def get_previous_source_words_with_tag(self, word_index, n_tags, tags=("NN", "NNS")):
        word_index_doc_pairs = self.get_previous_indices_with_tag(word_index, n_tags, tags)
        source_words = []
        current_doc = 0
        current_sentence = self
        for word_index, s_id in word_index_doc_pairs:
            if word_index == "N/A":
                source_words.append("N/A")
                continue
            elif s_id > current_doc:
                # jumping to sentence with s_id
                for _ in range(s_id - current_doc):
                    current_sentence = current_sentence.prev_sentence
                current_doc = s_id
            source_words.append(current_sentence.source_sentence[word_index])
        return source_words

    def get_previous_target_words_with_tag(self, word_index, n_tags, tags=("NN", "NNS")):
        word_index_doc_pairs = self.get_previous_indices_with_tag(word_index, n_tags, tags)
        source_words = []
        current_doc = 0
        current_sentence = self
        for word_index, s_id in word_index_doc_pairs:
            if word_index == "N/A":
                source_words.append(["N/A"])
                continue
            elif s_id > current_doc:
                # jumping to sentence with s_id
                for _ in range(s_id - current_doc):
                    current_sentence = current_sentence.prev_sentence
                current_doc = s_id
            # Not all source nouns are aligned to a target noun
            if word_index in current_sentence.source2target_alignments:
                target_indices = current_sentence.source2target_alignments[word_index]
                source_words.append([current_sentence.target_sentence[wid] for wid in target_indices])
            else:
                source_words.append(["N/A"])

        return source_words

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
            context = []
            # Add sentence start symbols when out-of-bounds
            for _ in range(max(0, left_context - source_indices[0])):
                context.append("<S>")

            for i in range(-left_context, 0):
                if source_indices[0] + i >= 0:
                    context.append(self.source_sentence[source_indices[0] + i])

            context.append([self.source_sentence[i] for i in source_indices])

            for i in range(1, right_context + 1):
                if source_indices[-1] + i < len(self.source_sentence):
                    context.append(self.source_sentence[source_indices[-1] + i])

            # Add sentence end symbols when out-of-bounds
            for _ in range(max(0, 1 + right_context + source_indices[-1] - len(self.source_sentence))):
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
