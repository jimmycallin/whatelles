#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Scores the output for the shared task of the DiscoMT workshop.
#
#
#  Last modified: Ferbuary 26, 2015
#
#
#  Use:
#     perl discoMT_scorer.pl [-y scores.yml] <GOLD_FILE> <PREDICTIONS_FILE>
#
#  Example use:
#     perl discoMT_scorer.pl gold.txt predicted.txt
#     perl discoMT_scorer.pl gold.txt predicted_simple.txt
#
#  Description:
#
#     The scorer calculates and outputs the following statistics:
#        (1) confusion matrix, which shows
#			- the count for each gold/predicted pair
#           - the sums for each row/column: -SUM-
#        (2) accuracy
#        (3) precision (P), recall (R), and F1-score for each label
#        (4) micro-averaged P, R, F1 (note that in our single-class classification problem, micro-P=R=F1=Acc)
#        (5) macro-averaged P, R, F1
#
#     The scoring is done two times:
#       (i)  using coarse-grained labels (ce, {cela+ça}, elle, elles, il, ils, {OTHER+on}).
#       (ii) using fine-grained labels   (ce, cela, elle, elles, il, ils, on, ça, OTHER).
#     
#     The official score is the macro-averaged F1-score for (ii).
#
#

use warnings;
use strict;
use utf8;
use Getopt::Std;

###################
###   GLOBALS   ###
###################

my %confMatrixCoarse   = ();
my @allLabelsCoarse    = ('  ce ', 'cela ', 'elle ', 'elles', '  il ', ' ils ', 'OTHER');
my %labelMappingCoarse = ('ce'=>'  ce ', 'cela'=>'cela ', 'elle'=>'elle ', 'elles'=>'elles', 'il'=>'  il ', 'ils'=>' ils ', 'on'=>'OTHER', 'ça'=>'cela ', 'OTHER'=>'OTHER');

my %confMatrixFine     = ();
my @allLabelsFine      = ('  ce ', 'cela ', 'elle ', 'elles', '  il ', ' ils ', '  on ', '  ça ', 'OTHER');
my %labelMappingFine   = ('ce'=>'  ce ', 'cela'=>'cela ', 'elle'=>'elle ', 'elles'=>'elles', 'il'=>'  il ', 'ils'=>' ils ', 'on'=>'  on ', 'ça'=>'  ça ', 'OTHER'=>'OTHER');


################
###   MAIN   ###
################

### 1. Check oparameters
our $opt_y;
getopts('y:');
die "Usage: $0 <GOLD_FILE> <PREDICTIONS_FILE>\n" if ($#ARGV != 1);
my $GOLD_FILE        = $ARGV[0];
my $PREDICTIONS_FILE = $ARGV[1];

### 2. Open the files
open GOLD, '<:encoding(UTF-8)', $GOLD_FILE or die "Error opening $GOLD_FILE!";
open PREDICTED, '<:encoding(UTF-8)', $PREDICTIONS_FILE or die "Error opening $PREDICTIONS_FILE!";

if ($opt_y) {
    open YAML, '>:encoding(UTF-8)', $opt_y;
    print YAML "filename: '$PREDICTIONS_FILE'\n";
    print YAML "scores:\n";
}

### 3. Collect the statistics
for (my $lineNo = 1; <GOLD>; $lineNo++) {
	
	# 3.1. Get the GOLD label
	# OTHER	le	There 's just no way of getting it right .	Il est impossible de de REPLACE_7 percevoir correctement .	0-0 1-1 1-3 2-2 3-2 4-2 5-3 5-4 6-6 7-5 8-7 9-8
        die "Line $lineNo: Wrong file format for $GOLD_FILE!" if (!/^([^\t]*)\t[^\t]*\t[^\t]+\t[^\t]+\t[^\t]+$/);
	my $goldLabel = $1;

	# 3.2. Get the PREDICTED label
	# ce	c'	There 's just no way of getting it right .	Il est impossible de de REPLACE_7 percevoir correctement .	0-0 1-1 1-3 2-2 3-2 4-2 5-3 5-4 6-6 7-5 8-7 9-8
        die "Line $lineNo: The file $PREDICTIONS_FILE is shorter!" if (!($_ = <PREDICTED>));
        die "Line $lineNo: Wrong file format for $PREDICTIONS_FILE!" if (!/^([^\t\n\r]*)/);
	my $predictedLabel = $1;

	# 3.3. Check the file formats
	if ($goldLabel eq '') {
		if ($predictedLabel eq '') {
			next;
		}
		else {
			die "Line $lineNo: The gold label is empty, but the predicted label is not: $predictedLabel";
		}
	}
	elsif ($predictedLabel eq '') {
		die "Line $lineNo: The predicted label is empty, but the gold label is not: $goldLabel";
	}

	die "Line $lineNo: Wrong file format for $GOLD_FILE: the gold label is '$goldLabel'" if ($goldLabel !~ /^(ce|cela|elle|elles|il|ils|on|ça|OTHER)( (ce|cela|elle|elles|il|ils|on|ça|OTHER))*$/);
	die "Line $lineNo: Wrong file format for $PREDICTIONS_FILE: the predicted label is '$predictedLabel'" if ($predictedLabel !~ /^(ce|cela|elle|elles|il|ils|on|ça|OTHER)( (ce|cela|elle|elles|il|ils|on|ça|OTHER))*$/);

	my @goldLabels      = split / /, $goldLabel;
	my @predictedLabels = split / /, $predictedLabel;
	die "Line $lineNo: Different number of labels in the gold and in the predictions file." if ($#goldLabels != $#predictedLabels);

	# 3.4. Update the statistics
	for (my $ind = 0; $ind <= $#goldLabels; $ind++) {
		my $gldLabel = $goldLabels[$ind];
		my $prdctdLabel = $predictedLabels[$ind];
		$confMatrixFine{$labelMappingFine{$prdctdLabel}}{$labelMappingFine{$gldLabel}}++;
		$confMatrixCoarse{$labelMappingCoarse{$prdctdLabel}}{$labelMappingCoarse{$gldLabel}}++;
	}

}

### 4. Coarse-grained evaluation
print "\n<<< I. COARSE EVALUATION >>>\n\n";
if ($opt_y) {
    print YAML "  coarse:\n";
}
&evaluate(\@allLabelsCoarse, \%confMatrixCoarse);

### 5. Fine-grained evaluation
print "\n<<< II. FINE-GRAINED EVALUATION >>>\n\n";
if ($opt_y) {
    print YAML "  fine:\n";
}
my ($officialScore, $accuracy) = &evaluate(\@allLabelsFine, \%confMatrixFine);

### 6. Output the official score
print "\n<<< III. OFFICIAL SCORE >>>\n";
printf "\nMACRO-averaged fine-grained F1: %6.2f%s", $officialScore, "%\n";

### 7. Close the files
close GOLD or die;
close PREDICTED or die;

### 8. Print a summary to the screen
print "$PREDICTIONS_FILE\t$officialScore\t$accuracy\n";


################
###   SUBS   ###
################

sub evaluate() {
	my ($allLabels, $confMatrix) = @_;

	### 0. Calculate the horizontal and vertical sums
	my %allLabelsProposed = ();
	my %allLabelsAnswer   = ();
	my ($cntCorrect, $cntTotal) = (0, 0);
	foreach my $labelGold (@{$allLabels}) {
		foreach my $labelProposed (@{$allLabels}) {
			$$confMatrix{$labelProposed}{$labelGold} = 0
				if (!defined($$confMatrix{$labelProposed}{$labelGold}));
			$allLabelsProposed{$labelProposed} += $$confMatrix{$labelProposed}{$labelGold};
			$allLabelsAnswer{$labelGold} += $$confMatrix{$labelProposed}{$labelGold};
			$cntTotal += $$confMatrix{$labelProposed}{$labelGold};
		}
		$cntCorrect += $$confMatrix{$labelGold}{$labelGold};
	}

	### 1. Print the confusion matrix heading
	print "Confusion matrix:\n";
	print "       ";
	foreach my $label (@{$allLabels}) {
		printf " %5s", $label;
	}
	print " <-- classified as\n";
	print "       +";
	foreach (@{$allLabels}) {
		print "------";
	}
	print "+ -SUM-\n";

	### 2. Print the rest of the confusion matrix
	my $freqCorrect = 0;
	foreach my $labelGold (@{$allLabels}) {

		### 2.1. Output the short relation label
		printf " %5s |", $labelGold;

		### 2.2. Output a row of the confusion matrix
		foreach my $labelProposed (@{$allLabels}) {
			printf "%5d ", $$confMatrix{$labelProposed}{$labelGold};
		}

		### 2.3. Output the horizontal sums
		printf "| %5d\n", $allLabelsAnswer{$labelGold};
	}
	print "       +";
	foreach (@{$allLabels}) {
		print "------";
	}
	print "+\n";
	
	### 3. Print the vertical sums
	print " -SUM- ";
	foreach my $labelProposed (@{$allLabels}) {
		printf "%5d ", $allLabelsProposed{$labelProposed};
	}
	print "\n\n";

	### 5. Output the accuracy
	my $accuracy = 100.0 * $cntCorrect / $cntTotal;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (calculated for the above confusion matrix) = ', $cntCorrect, '/', $cntTotal, ' = ', $accuracy, "\%\n";
        if ($opt_y) {
            print YAML "    total_acc: [$cntCorrect, $cntTotal]\n"
        }

	### 8. Output P, R, F1 for each relation
	my ($macroP, $macroR, $macroF1) = (0, 0, 0);
	my ($microCorrect, $microProposed, $microAnswer) = (0, 0, 0);
	print "\nResults for the individual labels:\n";
	foreach my $labelGold (@{$allLabels}) {

		### 8.3. Calculate P/R/F1
		my $P  = (0 == $allLabelsProposed{$labelGold}) ? 0
				: 100.0 * $$confMatrix{$labelGold}{$labelGold} / $allLabelsProposed{$labelGold};
		my $R  = (0 == $allLabelsAnswer{$labelGold}) ? 0
				: 100.0 * $$confMatrix{$labelGold}{$labelGold} / $allLabelsAnswer{$labelGold};
		my $F1 = (0 == $P + $R) ? 0 : 2 * $P * $R / ($P + $R);

		printf "%10s%s%5d%s%5d%s%6.2f", $labelGold,
			" :    P = ", $$confMatrix{$labelGold}{$labelGold}, '/', $allLabelsProposed{$labelGold}, ' = ', $P;

		printf "%s%5d%s%5d%s%6.2f%s%6.2f%s\n", 
		  	 "%     R = ", $$confMatrix{$labelGold}{$labelGold}, '/', $allLabelsAnswer{$labelGold},   ' = ', $R,
			 "%     F1 = ", $F1, '%';

                if ($opt_y) {
                    my $lbl = $labelGold;
                    $lbl =~ s/ //g;
                    printf YAML "    rel_%s: [%s, %s, %s]\n", $lbl, $$confMatrix{$labelGold}{$labelGold}, $allLabelsProposed{$labelGold}, $allLabelsAnswer{$labelGold};
                }

		### 8.5. Accumulate statistics for micro/macro-averaging
		$macroP  += $P;
		$macroR  += $R;
		$macroF1 += $F1;
		$microCorrect += $$confMatrix{$labelGold}{$labelGold};
		$microProposed += $allLabelsProposed{$labelGold};
		$microAnswer += $allLabelsAnswer{$labelGold};
	}

	### 9. Output the micro-averaged P, R, F1
	my $microP  = (0 == $microProposed)    ? 0 : 100.0 * $microCorrect / $microProposed;
	my $microR  = (0 == $microAnswer)      ? 0 : 100.0 * $microCorrect / $microAnswer;
	my $microF1 = (0 == $microP + $microR) ? 0 :   2.0 * $microP * $microR / ($microP + $microR);
	print "\nMicro-averaged result:\n";
	printf "%s%5d%s%5d%s%6.2f%s%5d%s%5d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrect, '/', $microProposed, ' = ', $microP,
		"%     R = ", $microCorrect, '/', $microAnswer, ' = ', $microR,
		"%     F1 = ", $microF1, '%';

	### 10. Output the macro-averaged P, R, F1
	my $distinctLabelsCnt = $#{$allLabels}+1; 

	$macroP  /= $distinctLabelsCnt; # first divide by the number of non-Other categories
	$macroR  /= $distinctLabelsCnt;
	$macroF1 /= $distinctLabelsCnt;
	print "\nMACRO-averaged result:\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP, "%\tR = ", $macroR, "%\tF1 = ", $macroF1, '%';

        if ($opt_y) {
            printf YAML "    macro_avg: %.3f\n", $macroF1;
        }

	### 11. Return the official score
	return ($macroF1, $accuracy);
}
