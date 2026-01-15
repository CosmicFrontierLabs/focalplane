#!/bin/bash
# Wrapper for deploy-fgs.sh - deploys to Orin Nano (PlayerOne camera)
# See deploy-fgs.sh for full documentation
exec "$(dirname "$0")/deploy-fgs.sh" orin "$@"
