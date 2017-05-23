#!/bin/bash
set -e


CLASSPATH="lib:lib/stanford-corenlp-3.7.0.jar:lib/stanford-corenlp-3.7.0-models.jar"
javac -cp $CLASSPATH lib/*.java
python2.7 scripts/preprocess-sick.py
python2.7 scripts/preprocess-sst.py