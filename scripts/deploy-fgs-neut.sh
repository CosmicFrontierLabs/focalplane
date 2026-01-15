#!/bin/bash
# Wrapper for deploy-fgs.sh - deploys to Neutralino (NSV455 camera)
# See deploy-fgs.sh for full documentation
exec "$(dirname "$0")/deploy-fgs.sh" neut "$@"
