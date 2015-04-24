#! /bin/bash

chunksize=50

if [ $# -ne 3 ]
then
	echo "$0 docstart prefix directory" 1>&2
	exit 1
fi

docstart=$1
prefix=$2
dirprefix=$3

mmax_skeleton=/Users/jimmy/dev/edu/mt/pronoun/resources/mmax-import/mmax-skeleton

if [ -e $dirprefix.001 ]
then
	echo "$dirprefix.001 already exists." 1>&2
	exit 1
fi

start=(`cat $docstart`)

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

for ((i=1; i<=${#start[@]}; i++))
do
	if [ $(($i % $chunksize)) -eq 1 ]
	then
		dir=$dirprefix.`printf '%03d' $(($i / $chunksize))`
		cp -r $mmax_skeleton $dir
	fi

	name=${prefix}_`printf '%07d' ${start[$i-1]}`

	{
		lineno=${start[$i-1]}
		echo -n "Document $i starts at line $lineno " 1>&2
		while [ $i -eq ${#start[@]} -o $lineno -lt "0${start[$i]}" ] && read -r
		do
			echo "$REPLY"
			let lineno++
		done
		echo "and ends at line $(($lineno - 1)) " 1>&2
	} |
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
