#!/bin/bash

overwrite=false
tools=""
for arg in $*; do
	case $arg in
		-f|--force)
			overwrite=true;;
		*)
			tools="$arg $tools"
			;;
	esac
done

for tool in $tools; do
	dir="data/docs/$tool"
	if [[ $overwrite == true ]]; then
		rm -rf "$dir"
	fi
	mkdir -p "$dir" || exit 1

	# topicmap
	cat <<EOF > "$dir/topicmap.xml"
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE map PUBLIC "-//OASIS//DTD DITA Map//EN" "map.dtd">
<map>
  <topicref href="name.md" format="markdown"/>
  <topicref href="release_notes.md" format="markdown"/>
  <topicref href="installation.md" format="markdown"/>
  <topicref href="tool.md" format="markdown"/>
  <topicref href="description.md" format="markdown"/>
  <topicref href="option.md" format="markdown"/>
  <topicref href="inputFormatDescription.md" format="markdown"/>
  <topicref href="parameters.md" format="markdown"/>
  <topicref href="outputFormatDescription.md" format="markdown"/>
  <topicref href="troubleshooting.xml"/>
  <topicref href="glossary.xml"/>
  <topicref href="authors.md" format="markdown"/>
  <topicref href="reporting.md" format="markdown"/>
  <topicref href="copyright.md" format="markdown"/>
</map>
EOF

	# name
	cat <<EOF > "$dir/name.md"
# $tool
EOF

	# simple description
	cat <<EOF > "$dir/tool.md"
# Tool $tool {#Tool .concept}
$(cat ocrd-tool.json | jq -r ".tools.\"$tool\".description")
EOF

	# parameters
	cat <<EOF > "$dir/parameters.md"
# Parameters {#parameters .reference}
The tool $tool accepts the following configuration parameters:
\`\`\`json
$(cat ocrd-tool.json | jq ".tools.\"$tool\".parameters")
\`\`\`
EOF

	# installation
	cat <<EOF > "$dir/installation.md"
# Installation of $tool {#installation .task}
1. (optional) Initialize virtualenv: \`python3 -m venv path/to/dir\`
2. Install ocrd_cis: \`make install\`
EOF

	# release notes
	cat <<EOF > "$dir/release_notes.md"
# Release notes
EOF

	# Authors
	cat<<EOF > "$dir/authors.md"
# Authors
1. Christoph Weber
2. Florian Fink
3. Robert Sachunsky
4. Tobias Englmeier
EOF

	# Reporting
	cat<<EOF > "$dir/reporting.md"
# Reporting
Reports any bugs/problems at the [issues page](https://github.com/cisocrgroup/ocrd_cis/issues)
EOF

	# Copyright
	echo "# License" > "$dir/copyright.md"
	cat LICENSE >> "$dir/copyright.md"

	# input format description
	if [[ ! -f "$dir/inputFormatDescription.md" ]]; then
		cat<<EOF > "$dir/inputFormatDescription.md"
# Input format {#inputFormatDescription .reference}
EOF
	fi

	# output format description
	if [[ ! -f "$dir/outputFormatDescription.md" ]]; then
		cat<<EOF > "$dir/outputFormatDescription.md"
# Output format {#outputFormatDescription .reference}
EOF
	fi

	# Troubleshooting
	if [[ ! -f "$dir/troubleshooting.xml" ]]; then
		cat<<EOF > "$dir/troubleshooting.xml"
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE troubleshooting
PUBLIC "-//OASIS//DTD DITA 1.3 Troubleshooting//EN" "troubleshooting.dtd">
<troubleshooting id="Troubleshooting">
    <title>Troubleshooting</title>
	<!--
    <troublebody>
        <condition>
            <title>Condition</title>
            <p></p>
        </condition>
        <troubleSolution>
            <cause>
                <title>Cause</title>
                <p></p>
            </cause>
            <remedy>
                <title>Remedy</title>
                <responsibleParty></responsibleParty>
                <steps>
                    <step>
                        <cmd></cmd>
                    </step>
                </steps>
            </remedy>
        </troubleSolution>
    </troublebody>
	-->
</troubleshooting>
EOF
	fi

	if [[ ! -f "$dir/glossary.xml" ]]; then
		cat<<EOF > "$dir/glossary.xml"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE glossgroup
PUBLIC "-//OASIS//DTD DITA Glossary Group//EN" "glossgroup.dtd">
<glossgroup id="Glossar">
    <title>Glossar</title>
	<!--
    <glossentry id="txtline">
        <glossterm>Textline</glossterm>
        <glossdef>A TextLine is a block of text without line break.
        </glossdef>
    </glossentry>
    <glossentry id="gt">
        <glossterm>Ground Truth</glossterm>
        <glossdef>Ground truth (GT) in the context of OCR-D are
        transcriptions, specific structure descriptions and word lists.
        These are essentially available in PAGE XML format in
        combination with the original image. Essential parts of
        the GT were created manually.
    </glossdef>
    -->
</glossgroup>
<!--
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE glossgroup
PUBLIC "-//OASIS//DTD DITA Glossary Group//EN" "glossgroup.dtd">
<glossgroup id="Glossar">
    <title>Glossar</title>
    <glossentry id="txtline">
        <glossterm>Textline</glossterm>
        <glossdef>A TextLine is a block of text without line break.
        </glossdef>
    </glossentry>
    <glossentry id="gt">
        <glossterm>Ground Truth</glossterm>
        <glossdef>Ground truth (GT) in the context of OCR-D are
        transcriptions, specific structure descriptions and word lists.
        These are essentially available in PAGE XML format in
        combination with the original image. Essential parts of
        the GT were created manually.
        </glossdef>
</glossgroup>
-->
EOF
	fi

	# generate description and options from README.md
	blockn=0
	ofile=""
	while read line; do
		if echo "$line" | grep $tool > /dev/null; then
			# echo "setting blockn=1"
			ofile="$dir/description.md"
			echo "# Description of $tool {#description .concept}" > "$ofile"
			blockn=1
		elif [[ $blockn == 1 ]] && [[ "$line" == "" ]]; then
			# echo "setting blockn=2"
			ofile="$dir/option.md"
			echo "# Options for $tool {#option .reference}" > "$ofile"
			blockn=2
		elif [[ $blockn == 2 ]] && [[ "$line" == "" ]]; then
			# echo "setting blockn=0"
			blockn=0
		elif [[ $blockn == 1 ]] || [[ $blockn == 2 ]]; then
			# echo "$blockn $line";
			echo "$line" >> "$ofile"
		fi
	done < README.md
done
