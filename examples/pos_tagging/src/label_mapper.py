from trapper.data.label_mapper import LabelMapper


@LabelMapper.register("conll2003_pos_tagging_example", constructor="from_labels")
class ExampleLabelMapperForPosTagging(LabelMapper):
    # Obtained by executing `dataset["train"].features["pos_tags"].feature.names`
    _LABELS = (
        '"', "''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT', 'EX',
        'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS',
        'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
        'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
        'WRB'
    )
