#!/bin/bash

# Install stanza
pip install stanza

# Download the English model for stanza
python -c "import stanza; stanza.download('en')"
