#!/bin/bash

for tool in ocrd-cis-wer; do
	dir="data/docs/$tool"
	mkdir -p "$dir" || exit 1
	cat <<EOF > "$dir/topicmap.xml"
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE map PUBLIC "-//OASIS//DTD DITA Map//EN" "map.dtd">
<map>
  <topicref href="tool.md" format="markdown"/>
  <topicref href="description.md" format="markdown"/>
  <topicref href="installation.md" format="markdown"/>
  <topicref href="option.md" format="markdown"/>
  <topicref href="parameters.md" format="markdown"/>
  <!--
  <topicref href="OutputFormatDescription.md" format="markdown"/>
  <topicref href="Resources.md" format="markdown"/>
  -->
</map>
EOF

	# tool
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
