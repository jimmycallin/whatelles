# whatelles
Repository for the cross-lingual pronoun classification task, a part of the Master's course in Machine Translation at Uppsala University. This was sent in to the shared task in cross-lingual pronoun classification as set up in the EMNLP 2015 Workshop on Discourse in Machine Translation, [DiscoMT](https://www.idiap.ch/workshop/DiscoMT). Please read [the paper](https://raw.githubusercontent.com/jimmycallin/whatelles/master/report/report.pdf) for a more detailed explanation.

## Dependencies
This project depends upon:
- Python >= 3.4
- theano
- numpy
- textblob

Please make sure you have these installed before running. I recommend using the excellent [Anaconda](https://store.continuum.io/cshop/anaconda/) distribution, since this simplifies the installation of Numpy quite a lot. 

## How to run

To get the final results as reported in the paper, please run:

    python test_final.py

If Python 3 isn't your default version, try:

    python3 test_final.py

(If you're using the recommended Anaconda distribution, you can easily create a new environment with Python 3 as the standard version.

    conda create -n py3 python=3 numpy
    source activate py3
    pip install theano

This creates and activates a new environment called `py3` with Python 3, along with Numpy and Theano installed. To deactivate, type `source deactivate`.)

When the training and testing is done, it should output the prediction file in `results/final.output.txt`.

To retrieve the evaluation results, use the discoMT_scorer.pl along with the gold standard file:

    perl discoMT_scorer.pl resources/test/discomt-gold/data.csv results/final.output.txt

## How to cite

If you're using any part of this in your own work, we would appreciate if you provided a citation:

    Jimmy Callin, Christian Hardmeier, and Jörg Tiedemann. 2015. Part-of-speech driven cross-lingual pronoun prediction with feed-forward neural networks. In Proceedings of the 2015 Workshop on Discourse in Machine Translation (DiscoMT 2015), Lisbon (Portugal).
