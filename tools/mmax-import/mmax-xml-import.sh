#! /bin/bash

if [ $# -ne 2 ]
then
	echo "$0 xml directory" 1>&2
	exit 1
fi

xmlcorpus=$1
dir=$2

mmax_skeleton=${DISCOMT_HOME:-..}/mmax-import/mmax-skeleton
xml=xml
#tokeniser="/usit/abel/u1/chm/WMT2013.en-fr/tokeniser/tokenizer.perl -l en"
tokeniser=cat

if [ -e $dir ]
then
	echo "$dir already exists." 1>&2
	exit 1
fi

cp -r "$mmax_skeleton" "$dir"

# This is needed because XML Starlet sometimes adds spurious newlines at the
# end of its output.
remove_last_if_empty='
	FNR > 1 {
		print lastline;
	}
	{
		lastline = $0;
	}
	END {
		if(lastline != "")
			print lastline;
	}'

docids=(`$xml sel -t -m "//doc" -v "@docid" -n $xmlcorpus | gawk "$remove_last_if_empty"`)

transform='
	BEGIN {
		widx = 1;
		sidx = 0;

		print "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>" >WORDS;
		print "<!DOCTYPE words SYSTEM \"words.dtd\">" >WORDS;
		print "<words>" >WORDS;

		print "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>" >SENTENCES;
		print "<!DOCTYPE markables SYSTEM \"markables.dtd\">" >SENTENCES;
		print "<markables xmlns=\"www.eml.org/NameSpaces/sentence\">" >SENTENCES;
	}

	{
		firstword = widx;
		for(i = 1; i <= NF; i++)
			print "<word id=\"word_" widx++ "\">" escape($i) "</word>" >WORDS;
		lastword = widx - 1;

		print "<markable mmax_level=\"sentence\" orderid=\"" sidx "\" id=\"markable_" sidx++ "\" span=\"word_" firstword "..word_" lastword "\" />" >SENTENCES;
	}

	END {
		print "</words>" >WORDS;
		print "</markables>" >SENTENCES;
	}

	function escape(s) {
		gsub(/&/, "\\&amp;", s);
		gsub(/</, "\\&lt;", s);
		gsub(/>/, "\\&gt;", s);
		return s;
	}'

for ((i=0; i<${#docids[@]}; i++))
do
	id=${docids[$i]}
	name="`printf '%03d' $i`_`echo $id | sed 's/[/\\]/_/g'`"
	echo "$id => $name" 1>&2

	$xml sel -t -m "//doc[@docid='$id']//seg" -v 'normalize-space(.)' -n $xmlcorpus |
		gawk "$remove_last_if_empty" |
		$xml unesc |
		$tokeniser 2>/dev/null |
		gawk -v WORDS=$dir/Basedata/${name}_words.xml -v SENTENCES=$dir/markables/${name}_sentence_level.xml "$transform"

	cat <<EOF >$dir/$name.mmax
<?xml version="1.0" encoding="UTF-8"?>
<mmax_project>
<words>${name}_words.xml</words>
<keyactions></keyactions>
<gestures></gestures>
</mmax_project>
EOF
done
