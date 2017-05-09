#!/bin/bash
set -e


CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"
javac -cp $CLASSPATH lib/*.java
python2.7 scripts/preprocess-sst.py